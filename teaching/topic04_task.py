"""Provided ML task machinery. No network/GPU work at import, no student answer."""
import math
from pathlib import Path
from typing import Any, Callable, Sequence, TypedDict
from teaching.topic04_types import PredictionRow,PredictionScores,ClientLike,TokenizerLike
from teaching.topic04_sse import ChatStream


class TaskCase(TypedDict):
    id: str
    label: str
    text: str


class TaskPolicy(TypedDict):
    name: str
    output_cap: int
    structured: bool
    system: str


class TaskObservation(PredictionRow, total=False):
    usage: dict[str, int] | None
    error_type: str
    request: dict[str, Any]
    rendered_input_ids: list[int]


class TaskProbe(TypedDict):
    contract: str
    status: str
    temperature: float
    output_cap: int
    structured: bool
    system: str
    observations: list[TaskObservation]

LABELS = ('billing', 'access', 'bug')
SYSTEM = ('Classify the support ticket. Reply with exactly one lowercase label: billing, access, or bug. '
          'billing = payments, invoices, refunds; access = login or account permissions; '
          'bug = software malfunction after successful login. No explanation.')


def sampling_distribution(logits: Sequence[float] = (2., 1., 0.), temperature: float = 1., top_p: float = 1.) -> list[float]:
    """Transform invented logits into a temperature/nucleus distribution.

    Parameters
    ----------
    logits : sequence of float
        Nonempty finite next-token logits, shape [vocabulary].
    temperature : float
        Nonnegative divisor; zero chooses the first maximum deterministically.
    top_p : float
        Cumulative mass threshold in (0,1], applied after temperature.

    Returns
    -------
    list[float]
        Renormalized probabilities in original vocabulary order; sum one.
        These are not measured Qwen logits or class correctness probabilities.

    Raises
    ------
    ValueError
        If inputs violate their finite-value/range contracts.
    """
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1 or not logits or not all(math.isfinite(v) for v in logits):
        raise ValueError('Finite logits, nonnegative temperature and 0 < top_p <= 1 required')
    if temperature == 0:
        best = max(range(len(logits)), key=lambda i: logits[i])
        return [float(i == best) for i in range(len(logits))]
    values = [math.exp((v-max(logits))/temperature) for v in logits]
    probs = [v/sum(values) for v in values]
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    kept, mass = [], 0.
    for i in order:
        kept.append(i); mass += probs[i]
        if mass >= top_p: break
    return [probs[i]/mass if i in kept else 0. for i in range(len(probs))]


def replay_response(records: Sequence[dict[str, Any] | str]) -> dict[str, Any]:
    """Reduce a finite teaching transcript with the supplied strict parser.

    Parameters
    ----------
    records : sequence of dict or str
        JSON envelopes or literal [DONE], in wire order.

    Returns
    -------
    dict
        content, finish_reason, done and status; partial text stays partial.
        All replay clocks are zero, so this makes no timing claim.
    """
    import json
    parser = ChatStream()
    for record in records:
        value = record if record == '[DONE]' else json.dumps(record, ensure_ascii=False)
        parser.feed(('data: '+value+'\n\n').encode('utf-8'), 0.)
    state = parser.finish()
    return {k: state[k] for k in ('content', 'finish_reason', 'done', 'status')}


def prediction_row(case_id: str, expected: str, content: str, status: str = 'complete', finish_reason: str | None = 'stop') -> TaskObservation:
    """Normalize one attempted ticket without extracting labels from prose.

    Parameters
    ----------
    case_id : str
        Stable ticket identifier.
    expected : str
        Gold label: billing, access or bug.
    content : str
        Literal observed response, possibly empty for failure.
    status : str
        complete, partial or failed lifecycle status.
    finish_reason : str or None
        Observed terminal reason, not an indication of correctness.

    Returns
    -------
    TaskObservation
        Gold/text/status and predicted label; None for invalid/incomplete text.
    """
    if expected not in LABELS or status not in ('complete', 'partial', 'failed') or not isinstance(content, str):
        raise ValueError('Invalid task observation')
    # Exact lowercase label after outer whitespace only; no substring extraction.
    candidate = content.strip()
    predicted = candidate if status == 'complete' and candidate in LABELS else None
    return dict(case_id=case_id, expected=expected, content=content, status=status,
                finish_reason=finish_reason, predicted=predicted)


def evaluation_fixture() -> list[TaskObservation]:
    """Create eight illustrative attempts, not a measured accuracy benchmark.

    Returns
    -------
    list[TaskObservation]
        Eight rows covering wrong labels, invalid format and lifecycle failure.
    """
    return [prediction_row('e1','billing','billing'), prediction_row('e2','billing','access'),
            prediction_row('e3','access','access'), prediction_row('e4','bug','bug',finish_reason='length'),
            prediction_row('e5','bug','billing because payment failed'),
            prediction_row('e6','access','access',status='partial'),
            prediction_row('e7','billing','',status='failed'), prediction_row('e8','bug','bug')]


def check_scores(score: Callable[[Sequence[PredictionRow]], PredictionScores]) -> int:
    """Check student scoring against empty, invalid and incomplete fixtures.

    Parameters
    ----------
    score : callable
        Maps attempted rows to all-attempt counts/rates and a [3,4] matrix.

    Returns
    -------
    int
        Number of passing fixtures; failures raise AssertionError.
    """
    fixtures = [([], (0,0,0,0)), (evaluation_fixture(), (8,4,5,6)),
                ([prediction_row('x','bug','access')], (1,0,1,1)),
                ([prediction_row('x','access','access',status='partial')], (1,0,0,0)),
                ([prediction_row('x','bug','bug',finish_reason='length')], (1,1,1,1)),
                ([prediction_row('x','billing','Billing')], (1,0,0,1))]
    for rows, counts in fixtures:
        result = score(rows)
        for key, count in zip(('attempts','correct','valid','complete'), counts):
            assert type(result[key]) is int and result[key] == count, (key,rows,result)
        n, correct, valid, complete = counts
        for key, expected in [('accuracy',correct/n if n else None),('valid_rate',valid/n if n else None),
                              ('completion_rate',complete/n if n else None)]:
            assert result[key] == expected, (key,rows,result)
        matrix = result['confusion']
        assert len(matrix)==3 and all(len(row)==4 for row in matrix)
        assert all(type(v) is int and v>=0 for row in matrix for v in row)
        assert sum(map(sum,matrix))==n
        for i,label in enumerate(LABELS):
            expected_row = [sum(r['expected']==label and r['predicted']==p for r in rows)
                            for p in (*LABELS,None)]
            assert matrix[i]==expected_row, (label,matrix,expected_row)
    return len(fixtures)


def live_task_probe(client: ClientLike, tokenizer: TokenizerLike, cases: Sequence[TaskCase], output_cap: int = 8,
                    structured: bool = False, save_partial: Callable[[TaskProbe], None] | None = None,
                    system: str = SYSTEM) -> TaskProbe:
    """One frozen setting, sequential task probe inside the owned service lifetime.

    The caller's absolute 120s bound covers the WHOLE auxiliary probe. Never retry.
    Failures remain rows; accuracy denominators must include every attempted ticket.

    Parameters
    ----------
    client : ClientLike
        Already connected local OpenAI-compatible client; retries disabled.
    tokenizer : TokenizerLike
        Pinned tokenizer used to count the complete rendered chat.
    cases : sequence of TaskCase
        Ordered unique tickets with id, text and gold label.
    output_cap : int
        Maximum selected output IDs, integer 1..16.
    structured : bool
        Restrict output to the three label strings when True.
    save_partial : callable or None
        Optional checkpoint callback after every attempted ticket.
    system : str
        Frozen nonempty system instruction, not student text.

    Returns
    -------
    TaskProbe
        Settings and all attempted observations, including failures and IDs.

    Raises
    ------
    TimeoutError
        The caller's whole-probe wall deadline expires; checkpoint then rethrow.
    ValueError
        Invalid settings, duplicate IDs or rendered context overflow.
    """
    if type(output_cap) is not int or not 1 <= output_cap <= 16 or type(structured) is not bool:
        raise ValueError('Task cap must be 1..16; structured must be bool')
    if not isinstance(system,str) or not system.strip():
        raise ValueError('Explicit nonempty system instruction required')
    observations = []
    if len({c['id'] for c in cases}) != len(cases) or any(c['label'] not in LABELS for c in cases):
        raise ValueError('Unique case IDs and known gold labels required')
    def record(status: str) -> TaskProbe:
        """Snapshot the accumulated probe without changing its population.

        Parameters
        ----------
        status : str
            running, complete or deadline-exceeded progress state.

        Returns
        -------
        TaskProbe
            Frozen settings and observations accumulated so far.
        """
        return dict(contract='topic04-task-probe-v1',status=status,temperature=0,
                    output_cap=output_cap,structured=structured,system=system,observations=observations)
    for case in cases:
        messages=[dict(role='system',content=system),dict(role='user',content=case['text'])]
        encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        ids=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
        if ids and isinstance(ids[0],list): ids=ids[0]
        if len(ids)+output_cap>2048: raise ValueError('Task request exceeds frozen context budget')
        request=dict(model='topic04',messages=messages,temperature=0,max_tokens=output_cap)
        if structured: request['extra_body']={'structured_outputs':{'choice':list(LABELS)}}
        try:
            answer=client.chat.completions.create(**request)
            choice=answer.choices[0]
            state='complete' if choice.finish_reason in ('stop','length') else 'failed'
            row=prediction_row(case['id'],case['label'],choice.message.content or '',
                               status=state,finish_reason=choice.finish_reason)
            row['usage']=answer.usage.model_dump() if answer.usage else None
        except TimeoutError:
            # Save the failed attempt, but never suppress the outer absolute deadline.
            row=prediction_row(case['id'],case['label'],'',status='failed')
            row.update(error_type='TimeoutError',request=request,rendered_input_ids=ids)
            observations.append(row)
            if save_partial is not None: save_partial(record('deadline-exceeded'))
            raise
        except Exception as exc:
            row=prediction_row(case['id'],case['label'],'',status='failed')
            row['error_type']=type(exc).__name__
        row.update(request=request,rendered_input_ids=ids)
        observations.append(row)
        if save_partial is not None: save_partial(record('running'))
    return record('complete')


def comparison_policies(challenger: str = 'prompt', output_cap: int = 8) -> list[TaskPolicy]:
    """Declare a matched baseline and one-factor challenger before measuring.

    Parameters
    ----------
    challenger : str
        prompt changes the system instruction; format changes the output constraint.
    output_cap : int
        Shared integer cap 1..16 selected IDs.

    Returns
    -------
    list[TaskPolicy]
        Two policies in baseline/challenger order, otherwise identical.
    """
    if challenger not in ('prompt', 'format') or type(output_cap) is not int or not 1 <= output_cap <= 16:
        raise ValueError('Choose prompt or format, with integer cap 1..16')
    clarified = SYSTEM + (' Decide by the problem that needs fixing. If login succeeds but a feature '
        'crashes, loses data or produces a wrong result, choose bug. Merely mentioning login '
        'does not mean access. Choose access when signing in or obtaining permission fails.')
    return [dict(name='baseline', output_cap=output_cap, structured=False, system=SYSTEM),
            dict(name=challenger, output_cap=output_cap, structured=challenger=='format',
                 system=clarified if challenger=='prompt' else SYSTEM)]


def baseline_and_recall(rows: Sequence[PredictionRow], score: Callable[[Sequence[PredictionRow]], PredictionScores]) -> dict[str, Any]:
    """Compute supplied all-attempt statistics and a constant-class control.

    Parameters
    ----------
    rows : sequence of PredictionRow
        All attempted tickets, not a filtered list of successful answers.
    score : callable
        Student scorer returning counts, rates and a [3,4] confusion matrix.

    Returns
    -------
    dict
        Constant-access correct count/accuracy, per-class recall and scores.
        Empty populations have None rates; absent true classes have None recall.
    """
    n = len(rows)
    return dict(constant_access_correct=sum(r['expected']=='access' for r in rows),
                constant_access_accuracy=sum(r['expected']=='access' for r in rows)/n if n else None,
                recall={label: (sum(r['expected']==label and r['predicted']==label for r in rows)
                                /sum(r['expected']==label for r in rows)
                                if any(r['expected']==label for r in rows) else None) for label in LABELS},
                scores=score(rows))


def freeze_selection(comparison: dict[str, Any], policies: Sequence[TaskPolicy], chosen: str,
                     explanation: str, path: str | Path) -> dict[str, Any]:
    """Save a calibration decision before access to held-out teaching cases.

    Parameters
    ----------
    comparison : dict
        Completed calibration comparison, including policies and raw probes.
    policies : sequence of TaskPolicy
        The two declared matched policies, in observation order.
    chosen : str
        Name of one observed policy.
    explanation : str
        Nonempty calibration-based justification; no held-out tuning.
    path : str or Path
        New decision file; existing evidence is never overwritten.

    Returns
    -------
    dict
        Payload with selected policy/comparison/explanation and its SHA-256.

    Raises
    ------
    ValueError
        Incomplete comparison or changed settings/population.
    FileExistsError
        The decision path already exists; preserve the old attempt.
    """
    import hashlib, json
    from pathlib import Path
    if comparison.get('split') != 'calibration' or comparison.get('status') != 'complete':
        raise ValueError('Completed calibration comparison required')
    names = [p['name'] for p in policies]
    if chosen not in names or not isinstance(explanation,str) or not explanation.strip():
        raise ValueError('Choose an observed policy and explain the calibration trade-off')
    if [r['policy'] for r in comparison['results']] != policies:
        raise ValueError('Policy changed after calibration')
    if len(policies)!=2 or policies!=comparison_policies(policies[1]['name'],policies[0]['output_cap']):
        raise ValueError('Declare exactly the matched baseline and one-factor challenger')
    for result in comparison['results']:
        policy,probe=result['policy'],result['probe']
        if probe.get('status')!='complete' or probe.get('temperature')!=0:
            raise ValueError('Incomplete or unmatched calibration probe')
        if any(probe.get(k)!=policy[k] for k in ('system','structured','output_cap')):
            raise ValueError('Observed probe settings differ from the declared policy')
        for row in probe['observations']:
            request=row['request']
            constraint={'structured_outputs':{'choice':list(LABELS)}} if policy['structured'] else None
            if (request.get('temperature')!=0 or request.get('max_tokens')!=policy['output_cap']
                    or request.get('model')!='topic04' or request.get('extra_body')!=constraint
                    or request['messages'][0]!=dict(role='system',content=policy['system'])):
                raise ValueError('A calibration request violates its declared control')
    populations = [[(r['case_id'],r['expected'],r['request']['messages'][-1]['content'])
                    for r in result['probe']['observations']] for result in comparison['results']]
    if len(populations)!=2 or not populations[0] or populations[0]!=populations[1]:
        raise ValueError('Both candidates must retain the same calibration attempts in order')
    payload=dict(contract='topic04-frozen-choice-v1', selected=policies[names.index(chosen)],
                 explanation=explanation.strip(), comparison=comparison)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    record=dict(payload=payload,sha256=digest)
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    # An old decision cannot silently become a new one after test results.
    if target.exists(): raise FileExistsError('New choice requires a new attempt directory')
    target.write_text(json.dumps(record,indent=2,ensure_ascii=False))
    return record


def read_frozen(path: str | Path) -> dict[str, Any]:
    """Load and verify a previously frozen decision.

    Parameters
    ----------
    path : str or Path
        Existing calibration decision JSON file.

    Returns
    -------
    dict
        Verified payload/digest; changed payloads raise ValueError.
    """
    import hashlib, json
    from pathlib import Path
    record=json.loads(Path(path).read_text())
    digest=hashlib.sha256(json.dumps(record['payload'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if record['sha256']!=digest or record['payload']['contract']!='topic04-frozen-choice-v1':
        raise ValueError('Frozen decision changed')
    return record


def heldout_cases(frozen: dict[str, Any]) -> list[TaskCase]:
    """Expose public teaching cases only after verifying a frozen decision.

    Parameters
    ----------
    frozen : dict
        Calibration payload and matching digest from read_frozen.

    Returns
    -------
    list[TaskCase]
        Six tickets, two per true class. This is procedural ordering, not
        cryptographic secrecy or a hidden grading population.
    """
    import hashlib, json
    digest=hashlib.sha256(json.dumps(frozen['payload'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if frozen['sha256']!=digest or frozen['payload']['contract']!='topic04-frozen-choice-v1':
        raise ValueError('Valid frozen decision required before held-out access')
    return [dict(id='s1',label='billing',text='The receipt for our payment has the wrong tax number.'),
            dict(id='s2',label='billing',text='A cancelled paid plan has still taken money from my card.'),
            dict(id='s3',label='access',text='The password reset link expired, so I cannot sign in.'),
            dict(id='s4',label='access',text='The administrator has not granted me permission to view the project.'),
            dict(id='s5',label='bug',text='Login works, but the application freezes whenever I sort the table.'),
            dict(id='s6',label='bug',text='After signing in, every exported PDF has missing pages.')]
