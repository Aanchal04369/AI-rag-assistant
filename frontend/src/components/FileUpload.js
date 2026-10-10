import React, { useRef, useState } from "react";
import { FileUp, UploadCloud, Loader2 } from "lucide-react";

export default function FileUpload({ onUpload, processing }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const choose = (file) => {
    if (file && file.type === "application/pdf") onUpload(file);
  };

  return (
    <div className="upload-card">
      <div
        className={`drop-zone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]); }}
        onClick={() => inputRef.current?.click()}
      >
        <input ref={inputRef} type="file" accept=".pdf,application/pdf" hidden
          onChange={(e) => choose(e.target.files[0])} />
        <div className="upload-icon"><UploadCloud size={34} /></div>
        <h2>Upload a research paper</h2>
        <p>Drag & drop a PDF here or click to browse</p>
        <button className="primary-button" type="button" onClick={(e) => {e.stopPropagation(); inputRef.current?.click();}}>
          <FileUp size={18}/> Choose PDF
        </button>
        <span className="muted">Text-based PDF • Maximum 16 MB</span>
      </div>
      {processing && <div className="processing"><Loader2 className="spin" size={18}/> Processing your document...</div>}
    </div>
  );
}
