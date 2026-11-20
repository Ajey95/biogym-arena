import { useEffect, useState } from 'react'
import { FlaskConical, Info, X } from 'lucide-react'
import {
  type Attempt, type CaseInfo, type Evaluation, type Job, type Room,
  type TrainingResult, getAttempt, getJob, getLatestEvaluation, getLatestTraining, getRooms, runEvaluation,
  sendAction, startAttempt, startTraining,
} from './api'
import { EvaluationScreen, MapScreen, RoomScreen, TrainingScreen } from './screens'

type Screen = 'map' | 'room' | 'training' | 'transfer' | 'evaluation'
type RoomId = 'sequence' | 'structure' | 'epitope'
const requiredRooms: RoomId[] = ['sequence', 'structure', 'epitope']

function App() {
  const [screen, setScreen] = useState<Screen>('map')
  const [roomId, setRoomId] = useState<RoomId>('sequence')
  const [rooms, setRooms] = useState<Room[]>([])
  const [activeCase, setActiveCase] = useState<CaseInfo | null>(null)
  const [attempt, setAttempt] = useState<Attempt | null>(null)
  const [recorded, setRecorded] = useState<Record<string, string>>(() => {
    try { return JSON.parse(localStorage.getItem('biogym-recorded') || '{}') } catch { return {} }
  })
  const [training, setTraining] = useState<TrainingResult | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getRooms().then(setRooms).catch((reason) => setError('Backend unavailable: ' + reason.message))
    Promise.all([getLatestTraining(), getLatestEvaluation()]).then(([latestTraining, latestEvaluation]) => {
      if (latestTraining) setTraining(latestTraining)
      if (latestTraining && latestEvaluation && latestEvaluation.trained_policy_id === latestTraining.trained.id) {
        setEvaluation(latestEvaluation)
      }
    }).catch(() => {})
  }, [])
  useEffect(() => {
    localStorage.setItem('biogym-recorded', JSON.stringify(recorded))
  }, [recorded])
  useEffect(() => {
    if (!jobId) return
    const timer = window.setInterval(async () => {
      try {
        const next = await getJob(jobId)
        setJob(next)
        if (next.status === 'COMPLETE' && next.result) {
          setTraining(next.result); setJobId(null)
        }
        if (next.status === 'FAILED' || next.status === 'INTERRUPTED') {
          setError(next.error || 'Training interrupted')
          setJobId(null)
        }
      } catch (reason) { setError(String(reason)); setJobId(null) }
    }, 1100)
    return () => window.clearInterval(timer)
  }, [jobId])

  const allRoomsRecorded = requiredRooms.every((id) => !!recorded[id])
  const currentRoom = rooms.find((room) => room.id === roomId)
  const transferCases = rooms.flatMap((room) => room.cases)
    .filter((item) => item.split === 'transfer' || item.split === 'transfer_live')

  function enterRoom(id: RoomId) {
    setRoomId(id); setScreen('room'); setAttempt(null); setActiveCase(null); setError(null)
  }
  function openTransfer() {
    if (!allRoomsRecorded) {
      setError('Record an attempt in each scientific room to open TransferArena.')
      return
    }
    setScreen('transfer'); setAttempt(null); setActiveCase(null); setError(null)
  }
  async function beginEpisode(item: CaseInfo) {
    setBusy(true); setError(null)
    try {
      const next = await startAttempt(item.id)
      setAttempt(next); setActiveCase(item)
    } catch (reason) { setError(String(reason)) } finally { setBusy(false) }
  }
  async function act(action: Record<string, unknown>) {
    if (!attempt) return
    setBusy(true); setError(null)
    try {
      const result = await sendAction(attempt.id, { action_id: crypto.randomUUID(), ...action })
      const next = await getAttempt(attempt.id)
      setAttempt(next)
      if (result.done) {
        const key = screen === 'transfer' ? 'transfer' : activeCase?.room
        if (key) setRecorded((old) => ({ ...old, [key]: attempt.id }))
      }
    } catch (reason) { setError(String(reason)) } finally { setBusy(false) }
  }
  async function train() {
    setBusy(true); setError(null)
    try {
      const response = await startTraining(400)
      setEvaluation(null)
      setJobId(response.job_id); setJob(null)
    } catch (reason) { setError(String(reason)) } finally { setBusy(false) }
  }
  async function evaluatePolicies() {
    if (!training) return
    setBusy(true); setError(null)
    try {
      setEvaluation(await runEvaluation(training.baseline.id, training.trained.id))
      setScreen('evaluation')
    } catch (reason) { setError(String(reason)) } finally { setBusy(false) }
  }
  async function inspectTrace(attemptId: string, caseId: string, targetRoom: string, split: string) {
    setBusy(true)
    try {
      const detail = await getAttempt(attemptId)
      setAttempt(detail)
      setActiveCase(rooms.flatMap((room) => room.cases).find((item) => item.id === caseId) || null)
      setRoomId(targetRoom as RoomId)
      setScreen(split === 'transfer' || split === 'transfer_live' ? 'transfer' : 'room')
    } catch (reason) { setError(String(reason)) } finally { setBusy(false) }
  }

  return <div className="app">
    <header className="topbar">
      <button className="brand" onClick={() => setScreen('map')} aria-label="BioGym Arena home">
        <span className="brand-mark"><FlaskConical size={21} /></span><span>BioGym <em>Arena</em></span>
      </button>
      <nav aria-label="Primary navigation">
        <button className={screen === 'map' || screen === 'room' ? 'active' : ''} onClick={() => setScreen('map')}>Laboratory</button>
        <button className={screen === 'training' ? 'active' : ''} onClick={() => setScreen('training')}>Policy Lab</button>
        <button className={screen === 'transfer' ? 'active' : ''} onClick={openTransfer}>TransferArena</button>
        <button className={screen === 'evaluation' ? 'active' : ''} onClick={() => setScreen('evaluation')}>Evaluation Hall</button>
      </nav>
      <div className="top-status"><span className="pulse" /> LOCAL LAB <span className="status-divider" /> 3 ROOMS</div>
    </header>
    <main>
      {screen === 'map' && <MapScreen rooms={rooms} recorded={recorded} onEnter={enterRoom} onTransfer={openTransfer} />}
      {(screen === 'room' || screen === 'transfer') && <RoomScreen
        isTransfer={screen === 'transfer'} room={currentRoom} roomId={roomId}
        cases={screen === 'transfer' ? transferCases : currentRoom?.cases.filter((item) => !item.split.startsWith('transfer')) || []}
        attempt={attempt} activeCase={activeCase} busy={busy}
        onBack={() => setScreen('map')} onStart={beginEpisode} onAction={act}
        onClose={() => { setAttempt(null); setActiveCase(null) }}
        onEvaluation={() => setScreen('evaluation')}
      />}
      {screen === 'training' && <TrainingScreen job={job} jobId={jobId} training={training}
        busy={busy} onTrain={train} onEvaluate={evaluatePolicies} />}
      {screen === 'evaluation' && <EvaluationScreen evaluation={evaluation}
        onTrain={() => setScreen('training')} onInspect={inspectTrace} />}
    </main>
    {error && <div className="error-toast" role="alert"><Info size={18} /><span>{error}</span>
      <button onClick={() => setError(null)} aria-label="Dismiss error"><X size={16} /></button></div>}
    <footer><span>BIOGYM ARENA · SCIENTIFIC AGENT LAB</span>
      <span>COMPUTATIONAL RESEARCH ONLY <span className="footer-separator">/</span> STARTER CASES ARE FIXTURES</span></footer>
  </div>
}

export default App
