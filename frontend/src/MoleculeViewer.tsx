import { useEffect, useRef, useState } from 'react'
import * as Mol from '3dmol'
import { Atom, X } from 'lucide-react'

export function MoleculeViewer({ url, structureId, onClose }: {
  url: string; structureId: string; onClose: () => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let active = true
    let viewer: ReturnType<typeof Mol.createViewer> | null = null
    fetch(url).then((response) => {
      if (!response.ok) throw new Error('Coordinate artifact unavailable')
      return response.text()
    }).then((pdb) => {
      if (!active || !host.current) return
      viewer = Mol.createViewer(host.current, { backgroundColor: '#17291F' })
      viewer.addModel(pdb, 'pdb')
      viewer.setStyle({}, { cartoon: { color: '#C2A68C' } })
      viewer.zoomTo()
      viewer.render()
    }).catch((reason) => { if (active) setError(String(reason)) })
    return () => { active = false; viewer?.clear() }
  }, [url])
  return <div className="molecule-overlay" role="dialog" aria-modal="true" aria-label={'3D structure ' + structureId}>
    <div className="molecule-dialog">
      <div className="molecule-heading"><span><Atom size={19} /> LIVE COORDINATE VIEW · {structureId}</span>
        <button onClick={onClose} aria-label="Close molecular viewer"><X size={20} /></button></div>
      <div className="molecule-canvas" ref={host}>{error && <div className="molecule-error">{error}</div>}</div>
      <p>Rendered from the recorded PDB coordinate artifact. A computational structure view is not experimental validation of an epitope or drug effect.</p>
    </div>
  </div>
}
