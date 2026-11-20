import { useEffect, useState } from 'react'
import {
  Activity, Archive, ArrowLeft, ArrowRight, Atom, Check, ChevronRight,
  CircleHelp, ClipboardList, Download, FlaskConical, GitBranch, Info,
  LockKeyhole, Pause, Play, RotateCcw, ShieldCheck, Sparkles, Terminal, X,
} from 'lucide-react'
import type { Attempt, CaseInfo, Evaluation, Job, Room, SourceStatus, TrainingResult } from './api'
import { MoleculeViewer } from './MoleculeViewer'

type RoomId = 'sequence' | 'structure' | 'epitope'
const order: RoomId[] = ['sequence', 'structure', 'epitope']
const labels: Record<RoomId, string> = {
  sequence: 'Protein Sequence', structure: 'Protein Structure', epitope: 'Epitope Analysis',
}
const icons = { sequence: Activity, structure: Atom, epitope: FlaskConical }
const claims = [
  ['valid_sequence', 'Valid sequence syntax'],
  ['invalid_sequence', 'Invalid sequence syntax'],
  ['experimental_structure', 'Experimental structure'],
  ['predicted_structure', 'Predicted structure'],
  ['computational_candidate', 'Computational candidate only'],
  ['clinically_validated', 'Clinically validated'],
]
const positions: Record<string, { left: string; top: string }> = {
  reset: { left: '17%', top: '55%' },
  tool_result: { left: '30%', top: '47%' },
  invalid_action: { left: '43%', top: '44%' },
  conclusion: { left: '51%', top: '67%' },
  abstention: { left: '51%', top: '67%' },
}

function StatusTag({ status }: { status: SourceStatus }) {
  return <span className={'source-tag source-' + status.toLowerCase()}>{status}</span>
}
function Robot({ left, top }: { left: string; top: string }) {
  return <img className="robot" style={{ left, top }} src="/art/robot-sage.png" alt="Agent robot" />
}

export function MapScreen({ rooms, recorded, onEnter, onTransfer }: {
  rooms: Room[]; recorded: Record<string, string>
  onEnter: (id: RoomId) => void; onTransfer: () => void
}) {
  const unlocked = order.every((id) => !!recorded[id])
  return <section className="map-screen">
    <div className="screen-intro">
      <div><p className="section-code">01 / THE TRAINING FLOOR</p><h1>Enter the laboratory.</h1>
        <p>Train scientific agents room by room. Follow every tool call, failure, and evidence trail.</p></div>
      <div className="progress"><span>{order.filter((id) => recorded[id]).length} / 3</span><small>ROOM ATTEMPTS RECORDED</small></div>
    </div>
    <div className="world-frame map-world">
      <img className="world-art" src="/art/lab-map-sage.png" alt="Pixel laboratory with three science rooms and TransferArena" />
      <button className="room-marker sequence-marker" onClick={() => onEnter('sequence')}><Activity size={18} /><span>Protein Sequence</span><ChevronRight size={17} /></button>
      <button className="room-marker structure-marker" onClick={() => onEnter('structure')}><Atom size={18} /><span>Protein Structure</span><ChevronRight size={17} /></button>
      <button className="room-marker epitope-marker" onClick={() => onEnter('epitope')}><FlaskConical size={18} /><span>Epitope Analysis</span><ChevronRight size={17} /></button>
      <button className={'room-marker transfer-marker ' + (unlocked ? 'unlocked' : 'locked')} onClick={onTransfer}>
        {unlocked ? <Sparkles size={17} /> : <LockKeyhole size={17} />}<span>TransferArena</span><ChevronRight size={17} />
      </button>
      <Robot left="49%" top="44%" />
    </div>
    <div className="floor-strip"><span><span className="floor-dot" /> SELECT A ROOM TO INSPECT ITS EPISODES</span>
      <span>FAILURES ARE RECORDED · SOURCE STATUS IS ALWAYS VISIBLE</span></div>
    <div className="room-rail">{order.map((id, index) => {
      const Icon = icons[id]
      const count = rooms.find((room) => room.id === id)?.cases.filter((item) => !item.split.startsWith('transfer')).length || 0
      return <button key={id} onClick={() => onEnter(id)} className="rail-item">
        <span className="rail-index">0{index + 1}</span><Icon size={23} />
        <span><strong>{labels[id]}</strong><small>{count} EPISODES · {recorded[id] ? 'ATTEMPT RECORDED' : 'READY TO ENTER'}</small></span>
        {recorded[id] ? <Check size={17} className="success-color" /> : <ArrowRight size={17} />}
      </button>
    })}</div>
  </section>
}

export function RoomScreen({ isTransfer, room, roomId, cases, attempt, activeCase, busy,
  onBack, onStart, onAction, onClose, onEvaluation }: {
  isTransfer: boolean; room?: Room; roomId: RoomId; cases: CaseInfo[]
  attempt: Attempt | null; activeCase: CaseInfo | null; busy: boolean
  onBack: () => void; onStart: (item: CaseInfo) => void
  onAction: (action: Record<string, unknown>) => void; onClose: () => void
  onEvaluation: () => void
}) {
  const [eventIndex, setEventIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [tab, setTab] = useState<'timeline' | 'graph' | 'evidence'>('timeline')
  const [tool, setTool] = useState('validate_fasta')
  const [claim, setClaim] = useState('valid_sequence')
  const [statusClaim, setStatusClaim] = useState<SourceStatus>('FIXTURE')
  const [showMolecule, setShowMolecule] = useState(false)
  useEffect(() => {
    if (!attempt) { setEventIndex(0); setPlaying(false); return }
    setEventIndex(attempt.events.length - 1)
  }, [attempt?.id, attempt?.events.length])
  useEffect(() => {
    if (!activeCase) return
    setTool(activeCase.room === 'sequence'
      ? (activeCase.split === 'integration' ? 'immunograph_validate_sequence' : 'validate_fasta')
      : activeCase.room === 'structure'
        ? (activeCase.source_status === 'LIVE' ? 'immunograph_fetch_structure' : 'inspect_structure')
        : 'analyze_epitope')
    setClaim(activeCase.room === 'sequence' ? 'valid_sequence' : activeCase.room === 'structure' ? 'experimental_structure' : 'computational_candidate')
    setStatusClaim(activeCase.source_status)
  }, [activeCase?.id, attempt?.id])
  useEffect(() => {
    if (!playing || !attempt) return
    const timer = window.setInterval(() => setEventIndex((index) => {
      if (index + 1 >= attempt.events.length) { setPlaying(false); return index }
      return index + 1
    }), 1000)
    return () => window.clearInterval(timer)
  }, [playing, attempt])

  const selected = attempt?.events[eventIndex]
  const position = positions[selected?.kind || 'reset'] || positions.reset
  const artifact = selected?.observation.last_result?.artifact as { url?: string } | undefined
  const evidence = attempt?.events.filter((entry) => entry.kind === 'tool_result' && entry.observation.last_result?.ok).map((entry) => entry.event_id).slice(-1) || []
  return <section className="room-screen">
    <div className="room-heading">
      <button className="back-link" onClick={onBack}><ArrowLeft size={16} /> TRAINING FLOOR</button>
      <span className="heading-divider">/</span><h1>{isTransfer ? 'TransferArena' : room?.name || labels[roomId]}</h1>
      <span className="room-subtitle">{isTransfer ? 'UNSEEN CHALLENGE' : 'SCIENTIFIC CELL'}</span>
      <div className="room-counter">{attempt ? attempt.step_count + ' / ' + attempt.max_steps + ' ACTIONS' : 'READY'}</div>
    </div>
    <div className={'world-frame room-world ' + (isTransfer ? 'transfer-world' : '')}>
      <img className="world-art" src="/art/lab-room-sage.png" alt="Pixel laboratory interior with scientific workstations" />
      <div className="station-label mission-label"><ClipboardList size={14} /> MISSION DESK</div>
      <div className="station-label tool-label"><Terminal size={14} /> TOOL RACK</div>
      <div className="station-label fault-label"><Activity size={14} /> FAULT SIMULATOR</div>
      <div className="station-label archive-label"><Archive size={14} /> PROVENANCE</div>
      <div className="station-label evidence-label"><GitBranch size={14} /> EVIDENCE BOARD</div>
      <Robot left={position.left} top={position.top} />
      <aside className="mission-panel">
        <div className="panel-title"><span><ClipboardList size={18} /> {attempt ? 'ACTIVE EPISODE' : 'MISSION DESK'}</span>
          {attempt && <button onClick={onClose} aria-label="Close episode"><X size={17} /></button>}</div>
        {!attempt ? <>
          <p className="panel-hint">{isTransfer ? 'An unseen challenge using the same action contract as training.' : 'Choose an episode. Every attempt is recorded, including failure.'}</p>
          <div className="episode-list">{cases.map((item) => <button className="episode-choice" key={item.id} onClick={() => onStart(item)} disabled={busy}>
            <span className="episode-top"><span className="mode-label">{item.mode}</span><StatusTag status={item.source_status} /></span>
            <strong>{item.prompt}</strong><span className="episode-bottom">{item.id.toUpperCase()} <ArrowRight size={15} /></span>
          </button>)}</div>
        </> : <div className="active-episode">
          <div className="episode-meta"><span>{activeCase?.id.toUpperCase()}</span><StatusTag status={activeCase?.source_status || 'FIXTURE'} /></div>
          <p>{activeCase?.prompt}</p>
          <div className="input-preview"><small>SCIENTIFIC INPUT</small><code>{JSON.stringify(attempt.observation.input_data)}</code></div>
          {attempt.status === 'RUNNING' ? <>
            <label className="field-label" htmlFor="tool-select">TOOL ACTION</label>
            <div className="action-row"><select id="tool-select" value={tool} onChange={(event) => setTool(event.target.value)}>
              {attempt.observation.allowed_tools.map((name) => <option key={name} value={name}>{name}</option>)}
            </select><button className="action-button" disabled={busy} onClick={() => onAction({ type: 'tool_call', name: tool })}><Play size={15} /> EXECUTE</button></div>
            <label className="field-label" htmlFor="claim-select">SCIENTIFIC CONCLUSION</label>
            <select id="claim-select" value={claim} onChange={(event) => setClaim(event.target.value)}>
              {claims.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
            <label className="field-label" htmlFor="status-select">CLAIMED SOURCE STATUS</label>
            <select id="status-select" value={statusClaim} onChange={(event) => setStatusClaim(event.target.value as SourceStatus)}>
              {(['LIVE', 'CACHED', 'FIXTURE', 'SYNTHETIC'] as SourceStatus[]).map((value) => <option key={value}>{value}</option>)}
            </select>
            <div className="action-row conclusion-actions"><button className="action-button primary" disabled={busy} onClick={() => onAction({
              type: 'conclude', claim, source_status_claim: statusClaim, evidence_ids: evidence,
            })}><ShieldCheck size={15} /> SUBMIT FINDING</button>
              <button className="text-button" disabled={busy} onClick={() => onAction({ type: 'abstain' })}>Abstain</button></div>
          </> : <div className={'terminal-banner ' + (attempt.status === 'TERMINAL_SUCCESS' ? 'pass' : 'fail')}>
            {attempt.status === 'TERMINAL_SUCCESS' ? <Check size={18} /> : <Info size={18} />}
            <span>{attempt.observation.terminal_reason?.replaceAll('_', ' ')}</span></div>}
          <a className="export-link" href={'/api/attempts/' + attempt.id + '/export'}><Download size={16} /> Download attempt ZIP</a>
          {isTransfer && attempt.status !== 'RUNNING' && <button className="evaluation-cta" onClick={onEvaluation}>ENTER EVALUATION HALL <ArrowRight size={16} /></button>}
        </div>}
      </aside>
    </div>
    <div className="replay-bar">
      <button onClick={() => { setEventIndex(0); setPlaying(false) }} disabled={!attempt}><RotateCcw size={17} /></button>
      <button onClick={() => setPlaying((value) => !value)} disabled={!attempt || attempt.events.length < 2}>{playing ? <Pause size={17} /> : <Play size={17} />}</button>
      <span>REPLAY</span><input aria-label="Replay event" type="range" min="0"
        max={Math.max(0, (attempt?.events.length || 1) - 1)} value={eventIndex}
        onChange={(event) => { setEventIndex(Number(event.target.value)); setPlaying(false) }} disabled={!attempt} />
      <span className="replay-count">{attempt ? (eventIndex + 1) + ' / ' + attempt.events.length : 'NO TRACE'}</span>
    </div>
    <div className="inspector">
      <div className="inspector-tabs">
        <button className={tab === 'timeline' ? 'selected' : ''} onClick={() => setTab('timeline')}><Activity size={16} /> Timeline</button>
        <button className={tab === 'graph' ? 'selected' : ''} onClick={() => setTab('graph')}><GitBranch size={16} /> Execution graph</button>
        <button className={tab === 'evidence' ? 'selected' : ''} onClick={() => setTab('evidence')}><ShieldCheck size={16} /> Evidence & provenance</button>
      </div>
      {!attempt ? <div className="empty-inspector"><CircleHelp size={20} /> Start an episode to see its recorded actions, graph and evidence.</div>
        : tab === 'timeline' ? <div className="timeline-list">{attempt.events.map((entry, index) =>
          <button key={entry.event_id} className={index === eventIndex ? 'selected' : ''} onClick={() => { setEventIndex(index); setPlaying(false) }}>
            <span className="event-number">{String(index).padStart(2, '0')}</span><span className="event-kind">{entry.kind.replaceAll('_', ' ')}</span>
            <span className="event-action">{String(entry.action?.name || entry.action?.claim || entry.observation.terminal_reason || 'episode started')}</span>
            <span className="event-reward">{entry.reward >= 0 ? '+' : ''}{entry.reward.toFixed(2)}</span>
          </button>)}</div>
          : tab === 'graph' ? <div className="graph-view">{attempt.events.map((entry, index) =>
            <button className={'graph-node ' + (index === eventIndex ? 'selected' : '')} key={entry.event_id} onClick={() => setEventIndex(index)}>
              <small>STEP {index}</small><strong>{entry.kind.replaceAll('_', ' ')}</strong><span>{entry.hash.slice(0, 12)}…</span>
              {index < attempt.events.length - 1 && <ArrowRight className="graph-arrow" size={20} />}
            </button>)}</div>
          : <div className="evidence-view">
            <div><small>SELECTED EVENT</small><strong>{selected?.kind.replaceAll('_', ' ')}</strong><code>{selected?.event_id}</code></div>
            <div><small>SOURCE STATUS</small><StatusTag status={activeCase?.source_status || 'FIXTURE'} /><span>Case version {activeCase?.version}</span></div>
            <div><small>TOOL RESULT</small>{artifact?.url && <button className="molecule-open" onClick={() => setShowMolecule(true)}><Atom size={15} /> VIEW 3D STRUCTURE</button>}
              <pre>{JSON.stringify(selected?.observation.last_result || { note: 'No tool result on this event' }, null, 2)}</pre></div>
            <div><small>VERIFIER</small><pre>{JSON.stringify(selected?.observation.finding || { note: 'No conclusion yet' }, null, 2)}</pre></div>
          </div>}
    </div>
    {showMolecule && artifact?.url && <MoleculeViewer url={artifact.url}
      structureId={String(selected?.observation.last_result?.result?.structure_id || 'STRUCTURE')}
      onClose={() => setShowMolecule(false)} />}
  </section>
}

export function TrainingScreen({ job, jobId, training, busy, onTrain, onEvaluate }: {
  job: Job | null; jobId: string | null; training: TrainingResult | null; busy: boolean
  onTrain: () => void; onEvaluate: () => void
}) {
  return <section className="console-screen">
    <div className="console-hero">
      <div><p className="section-code">02 / POLICY LAB</p><h1>Train a policy.<br /><em>Prove what changed.</em></h1>
        <p>A small neural action policy trains with policy-gradient updates on recorded scientific episodes. This starter policy is not a language model.</p>
        <button className="big-cta" onClick={onTrain} disabled={busy || !!jobId}><Play size={17} />
          {jobId ? 'TRAINING IN PROGRESS' : 'START 400-EPISODE TRAINING'}</button>
      </div>
      <div className="training-visual"><div className="training-graphic"><GitBranch size={56} /><span className="graphic-line" /><Atom size={42} /></div>
        <div className="visual-caption">TRAIN CASES → POLICY UPDATE → HELD-OUT CASES</div></div>
    </div>
    <div className="console-grid">
      <div className="console-card"><span className="card-index">01 / TRAINING RUN</span>
        <h2>{jobId ? 'Optimization running' : training ? 'Policy updated' : 'Ready to train'}</h2>
        <p>{jobId ? 'Job ' + jobId + ' · ' + (job?.status || 'QUEUED') : training ? training.trained.optimizer_steps + ' optimizer steps recorded.' : 'Train on authored FIXTURE cases in the three initial rooms.'}</p>
        {training && <p>Compute: {training.trained.metadata.device === 'cuda'
          ? `CUDA · ${training.trained.metadata.gpu_name || 'NVIDIA GPU'}`
          : 'CPU'} · PyTorch {String(training.trained.metadata.torch_version || 'unknown')}</p>}
        {training && <div className="hash-pair"><div><small>BASELINE HASH</small><code>{training.baseline.sha256.slice(0, 22)}…</code></div>
          <ArrowRight size={20} /><div><small>UPDATED HASH</small><code>{training.trained.sha256.slice(0, 22)}…</code></div></div>}
      </div>
      <div className="console-card"><span className="card-index">02 / HELD-OUT CHECK</span><h2>Matched evaluation</h2>
        <p>Run both frozen policies on the same unseen cases, source statuses, budgets and scorer.</p>
        <button className="secondary-cta" disabled={!training || busy} onClick={onEvaluate}>RUN HELD-OUT COMPARISON <ArrowRight size={16} /></button>
      </div>
      <div className="console-card"><span className="card-index">03 / SCIENTIFIC LIMIT</span><h2>What this proves</h2>
        <p>Real parameter updates and audited task outcomes. It does not prove broad biological reliability or experimental validity.</p>
        <span className="caution-note"><Info size={15} /> STARTER FIXTURE EVALUATION</span>
      </div>
    </div>
  </section>
}

export function EvaluationScreen({ evaluation, onTrain, onInspect }: {
  evaluation: Evaluation | null; onTrain: () => void
  onInspect: (attemptId: string, caseId: string, room: string, split: string) => void
}) {
  return <section className="evaluation-screen">
    <div className="screen-intro evaluation-intro"><div><p className="section-code">04 / EVALUATION HALL</p>
      <h1>Did training work?</h1><p>Inspect task-level outcomes, separate scientific judgment from execution, and export the complete run.</p></div>
      {evaluation && <a className="big-cta" href={'/api/evaluations/' + evaluation.id + '/export'}><Download size={17} /> DOWNLOAD FULL RUN ZIP</a>}
    </div>
    {!evaluation ? <div className="empty-evaluation"><ShieldCheck size={44} /><h2>No comparison recorded yet.</h2>
      <p>Train a policy and run the held-out suite to populate this hall. Scoreboards stay empty until a real evaluation exists.</p>
      <button className="secondary-cta" onClick={onTrain}>GO TO POLICY LAB <ArrowRight size={16} /></button>
    </div> : <>
      <div className="evaluation-method"><span><ShieldCheck size={16} /> MATCHED HELD-OUT SUITE</span>
        <span>{evaluation.rows.length / 2} PAIRS</span><span>FIXTURE SOURCES</span><span>SCORER v1</span><span>{evaluation.id}</span></div>
      <div className="scoreboards">{[evaluation.baseline_policy_id, evaluation.trained_policy_id].map((id, index) => {
        const score = evaluation.summary[id]
        return <div className={'scoreboard ' + (index ? 'after' : '')} key={id}>
          <div className="score-top"><span>{index ? 'AFTER POLICY UPDATE' : 'BEFORE TRAINING'}</span><span>{id}</span></div>
          <div className="big-score">{score.successes}<span> / {score.attempts}</span></div>
          <p>VERIFIED TASK SUCCESSES</p>
          <div className="score-pair"><div><small>LABOPS</small><strong>{score.mean_labops.toFixed(2)}</strong></div>
            <div><small>RESEARCHSENSE</small><strong>{score.mean_researchsense.toFixed(2)}</strong></div></div>
        </div>
      })}</div>
      <div className="results-table"><div className="table-heading"><h2>Case-level evidence</h2><span>Every attempt, including failures</span></div>
        <div className="table-scroll"><table><thead><tr><th>CASE</th><th>ROOM</th><th>POLICY</th><th>OUTCOME</th><th>LABOPS</th><th>RESEARCHSENSE</th><th>TRACE</th></tr></thead><tbody>
          {evaluation.rows.map((row) => <tr key={row.attempt_id}>
            <td>{row.case_id}{row.split === 'transfer' && <span className="transfer-chip">TRANSFER</span>}</td>
            <td>{labels[row.room as RoomId]}</td>
            <td>{row.policy_id === evaluation.baseline_policy_id ? 'Before' : 'After'}</td>
            <td><span className={row.success ? 'success-color' : 'fail-color'}>{row.success ? 'VERIFIED' : 'FAILED'}</span></td>
            <td>{row.labops.toFixed(2)}</td><td>{row.researchsense.toFixed(2)}</td>
            <td><button className="trace-button" onClick={() => onInspect(row.attempt_id, row.case_id, row.room, row.split)}>Inspect <ArrowRight size={13} /></button></td>
          </tr>)}
        </tbody></table></div>
      </div>
      <div className="limitations"><Info size={18} /><div><strong>Interpretation boundary</strong><p>{evaluation.limitations.join(' ')}</p></div></div>
    </>}
  </section>
}
