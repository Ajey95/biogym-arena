export type SourceStatus = 'LIVE' | 'CACHED' | 'FIXTURE' | 'SYNTHETIC'
export type CaseInfo = {
  id: string; room: string; mode: string; split: string; prompt: string
  source_status: SourceStatus; max_steps: number; version: string
}
export type Room = { id: string; name: string; station: string; cases: CaseInfo[] }
export type Observation = {
  phase: string; room: string; mode: string; prompt: string
  input_data: Record<string, unknown>; allowed_tools: string[]
  source_status: SourceStatus; last_result: Record<string, any> | null
  remaining_steps: number; finding?: Record<string, unknown>; terminal_reason?: string
}
export type Event = {
  event_id: string; sequence: number; kind: string; action_id: string | null
  action: Record<string, any> | null; observation: Observation
  reward: number; labops: number; researchsense: number; outcome: number
  hash: string; created_at: string
}
export type Attempt = {
  id: string; case_id: string; policy_id: string; status: string
  step_count: number; max_steps: number; observation: Observation; events: Event[]
}
export type StepResult = { done: boolean; event_id: string; reward: number; observation: Observation }
export type PolicyRecord = {
  id: string; sha256: string; parent_id: string | null; optimizer_steps: number
  metadata: Record<string, unknown>
}
export type TrainingResult = { baseline: PolicyRecord; trained: PolicyRecord }
export type Job = {
  id: string; kind: string; status: string; payload: Record<string, unknown>
  result: TrainingResult | null; error: string | null
}
export type Evaluation = {
  id: string; label: string; baseline_policy_id: string; trained_policy_id: string
  rows: { case_id: string; split: string; room: string; policy_id: string
    attempt_id: string; success: boolean; reward: number; labops: number
    researchsense: number; source_status: SourceStatus; terminal_reason: string }[]
  summary: Record<string, { successes: number; attempts: number; success_rate: number
    mean_reward: number; mean_labops: number; mean_researchsense: number }>
  limitations: string[]
}
export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail || response.statusText)
  }
  return response.json() as Promise<T>
}
export const getRooms = () => request<Room[]>('/api/rooms')
export const startAttempt = (caseId: string) => request<Attempt>('/api/attempts', {
  method: 'POST', body: JSON.stringify({ case_id: caseId }),
})
export const getAttempt = (id: string) => request<Attempt>('/api/attempts/' + id)
export const sendAction = (id: string, action: Record<string, unknown>) =>
  request<StepResult>('/api/attempts/' + id + '/actions', {
    method: 'POST', body: JSON.stringify(action),
  })
export const startTraining = (episodes: number) => request<{ job_id: string; status: string }>('/api/train', {
  method: 'POST', body: JSON.stringify({ episodes }),
})
export const getJob = (id: string) => request<Job>('/api/jobs/' + id)
export const getLatestTraining = () => request<TrainingResult | null>('/api/training/latest')
export const getLatestEvaluation = () => request<Evaluation | null>('/api/evaluation/latest')
export const runEvaluation = (baselineId: string, trainedId: string) =>
  request<Evaluation>('/api/evaluations', {
    method: 'POST',
    body: JSON.stringify({ baseline_policy_id: baselineId, trained_policy_id: trainedId, repeats: 1 }),
  })
