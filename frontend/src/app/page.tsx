"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Activity, MessageSquare } from "lucide-react";

export default function Home() {
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [statusText, setStatusText] = useState("Ready to converse.");
  const [botMessage, setBotMessage] = useState("");
  const [latencyMs, setLatencyMs] = useState<number>(0);

  // 🔥 Speed Stage: Locally-aggregated Transcription States
  const [currentTranscript, setCurrentTranscript] = useState("");
  const [fullTranscript, setFullTranscript] = useState<string[]>([]);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const isPlayingRef = useRef<boolean>(false);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const userIdRef = useRef<string>("");
  const lastStopReasonRef = useRef<"silence" | "forced">("silence");
  const isProcessingRef = useRef<boolean>(false);
  const pendingChunkRef = useRef<{ blob: Blob; stopReason: "silence" | "forced" } | null>(null);
  const requestIdRef = useRef<number>(0);
  const latestAppliedRequestIdRef = useRef<number>(0);

  useEffect(() => {
    const storageKey = "thymotalk_user_id";
    const existing = window.localStorage.getItem(storageKey);
    if (existing) {
      userIdRef.current = existing;
      return;
    }
    const generated = `user_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    window.localStorage.setItem(storageKey, generated);
    userIdRef.current = generated;
  }, []);

  const speakTextFallback = (text: string, isUrgent: boolean) => {
    if (!text || typeof window === "undefined" || !window.speechSynthesis) return;
    const speak = () => {
      if (isPlayingRef.current && !isUrgent) return;
      if (isUrgent) window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.onstart = () => { isPlayingRef.current = true; };
      utterance.onend = () => { isPlayingRef.current = false; };
      utterance.onerror = () => { isPlayingRef.current = false; };
      window.speechSynthesis.speak(utterance);
    };
    if (!isUrgent && analyserRef.current) {
      const poll = () => {
        if (!analyserRef.current || !isSessionActive) return;
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        const averageVolume = dataArray.reduce((p, c) => p + c, 0) / dataArray.length;
        if (averageVolume < 10) speak();
        else setTimeout(poll, 100);
      };
      poll();
      return;
    }
    speak();
  };

  const processAudioChunk = async (blob: Blob, stopReason: "silence" | "forced") => {
    if (blob.size === 0) return;
    if (isProcessingRef.current) {
      pendingChunkRef.current = { blob, stopReason };
      return;
    }

    isProcessingRef.current = true;
    const requestId = ++requestIdRef.current;

    const formData = new FormData();
    formData.append("file", blob, "chunk.webm");
    formData.append("user_id", userIdRef.current || "default_user");

    try {
      const requestStart = performance.now();

      // 🔥 Speed Stage: Optimized API Call (Async Parallel Backend)
      const response = await fetch("http://localhost:8000/analyze_chunk", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Backend connection failed");

      const data = await response.json();

      if (requestId >= latestAppliedRequestIdRef.current) {
        latestAppliedRequestIdRef.current = requestId;
        setLatencyMs(Math.round(performance.now() - requestStart));

        // 1. Technical Emotion status
        if (data.emotion) setStatusText(`Feeling: ${data.emotion}`);

        // 2. Human response bubble
        if (data.response) {
          setBotMessage(data.response);
          if (data.interrupt) speakTextFallback(data.response, data.urgent || false);
        }

        // 3. Transcript Accumulator (Stage Speed)
        // Backend now returns only current chunk to allow frontend reset-control
        if (data.transcript && data.transcript.trim().length > 0) {
          setCurrentTranscript(data.transcript);
          setFullTranscript((prev) => {
            const lastLine = prev[prev.length - 1];
            // Simple check to avoid appending exact duplicate chunks
            if (lastLine === data.transcript) return prev;
            return [...prev, data.transcript];
          });
        }
      }
    } catch (err) {
      console.error("Speed Processing Error:", err);
    } finally {
      isProcessingRef.current = false;
      if (pendingChunkRef.current) {
        const next = pendingChunkRef.current;
        pendingChunkRef.current = null;
        processAudioChunk(next.blob, next.stopReason);
      }
    }
  };

  const startSession = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      // 🚀 SPEED STAGE: RESET LOGIC 🔥
      // Every new session starts with a clean slate
      setFullTranscript([]);
      setBotMessage("");
      setCurrentTranscript("");
      setStatusText("Listening...");
      setLatencyMs(0);

      let isVoiceActive = false;
      let silenceStart = Date.now();
      let lastChunkStart = Date.now();
      const SILENCE_THRESHOLD = 5;
      const SILENCE_DURATION = 500;

      const captureLoop = () => {
        if (!analyserRef.current || !intervalRef.current) return;
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        const averageVolume = dataArray.reduce((p, c) => p + c, 0) / dataArray.length;
        const isNowActive = averageVolume > SILENCE_THRESHOLD;
        if (isNowActive) { isVoiceActive = true; silenceStart = Date.now(); }
        const isSilentLongEnough = isVoiceActive && (Date.now() - silenceStart > SILENCE_DURATION);
        const isTakingTooLong = Date.now() - lastChunkStart > 2000;
        if ((isSilentLongEnough || isTakingTooLong) && mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
          isVoiceActive = false;
          lastChunkStart = Date.now();
          lastStopReasonRef.current = isSilentLongEnough ? "silence" : "forced";
          mediaRecorderRef.current.stop();
        }
        requestAnimationFrame(captureLoop);
      };

      const createRecorder = () => {
        const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        recorder.ondataavailable = (e) => { if (e.data.size > 0) processAudioChunk(e.data, lastStopReasonRef.current); };
        recorder.onstop = () => { if (intervalRef.current) { const next = createRecorder(); mediaRecorderRef.current = next; next.start(); } };
        return recorder;
      };

      const initialRecorder = createRecorder();
      mediaRecorderRef.current = initialRecorder;
      initialRecorder.start();

      intervalRef.current = setInterval(() => { }, 1000);
      requestAnimationFrame(captureLoop);

      setIsSessionActive(true);
    } catch (err) {
      setStatusText("Mic access error.");
    }
  };

  const stopSession = () => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    if (mediaRecorderRef.current) {
      if (mediaRecorderRef.current.state !== "inactive") mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
      mediaRecorderRef.current = null;
    }
    if (audioContextRef.current) { audioContextRef.current.close(); audioContextRef.current = null; analyserRef.current = null; }
    if (currentAudioRef.current) { currentAudioRef.current.pause(); currentAudioRef.current = null; }
    isPlayingRef.current = false;
    setIsSessionActive(false);
    setStatusText("Session ended.");
    window.speechSynthesis.cancel();
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-[#FAFAFA] text-[#0F172A] selection:bg-[#D4AF37]">
      <div className="max-w-4xl w-full flex flex-col items-center space-y-12">

        {/* Header (UI preserved) */}
        <div className="text-center space-y-3">
          <h1 className="text-4xl md:text-5xl font-light tracking-wide">
            THYMOTALK<span className="font-semibold text-[#D4AF37]">.AI</span>
          </h1>
          <p className="text-[#0F172A]/40 uppercase text-[10px] tracking-widest font-medium">
            Intelligence • Analysis • Transcription
          </p>
        </div>

        {/* Action & Stats Row (UI preserved) */}
        <div className="flex flex-col md:flex-row items-center justify-center gap-12 w-full">
          <div className="flex flex-col items-center space-y-6">
            <button
              onClick={isSessionActive ? stopSession : startSession}
              className={`relative flex items-center justify-center w-36 h-36 rounded-full transition-all duration-700 ease-in-out border shadow-lg ${isSessionActive ? "bg-[#D4AF37] border-transparent scale-105" : "bg-white border-black/5 hover:border-[#D4AF37]/40"
                }`}
            >
              {isSessionActive ? <Activity className="w-14 h-14 text-white animate-pulse" /> : <Mic className="w-14 h-14 text-[#D4AF37]" strokeWidth={1} />}
            </button>
            <div className="flex flex-col items-center space-y-1">
              <span className={`text-[10px] font-bold uppercase tracking-widest ${isSessionActive ? "text-green-500" : "text-gray-400"}`}>
                {isSessionActive ? "ACTIVE" : "IDLE"}
              </span>
              <span className="text-[9px] text-black/20 font-mono">LATENCY: {latencyMs}ms</span>
            </div>
          </div>

          <div className="flex-1 w-full md:max-w-md bg-white border border-black/5 rounded-2xl p-6 shadow-sm min-h-[144px] flex flex-col">
            <div className="flex items-center space-x-2 text-[#D4AF37] mb-3">
              <MessageSquare className="w-4 h-4" />
              <span className="text-[10px] font-bold uppercase tracking-tighter">Real-time Transcript</span>
            </div>
            <div className="flex-1 overflow-y-auto max-h-[120px] pr-2 custom-scrollbar">
              {fullTranscript.length > 0 ? (
                <div className="space-y-2">
                  {fullTranscript.map((line, idx) => (
                    <p key={idx} className={`text-sm leading-relaxed ${idx === fullTranscript.length - 1 ? "text-black font-normal" : "text-black/30"}`}>
                      {line}
                    </p>
                  ))}
                </div>
              ) : (
                <p className="text-[#0F172A]/10 text-sm italic">
                  {isSessionActive ? "Waiting for speech..." : "Transcript will appear here"}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Emotion Display Area (UI preserved) */}
        <div className="w-full text-center space-y-6">
          <div className="h-6">
            <span className="text-[10px] text-[#D4AF37] font-bold uppercase tracking-[0.3em]">{statusText}</span>
          </div>
          <div className="min-h-[100px] flex items-center justify-center p-8 bg-black/[0.01] rounded-3xl border border-dashed border-black/10">
            {botMessage ? (
              <p className="text-3xl md:text-4xl font-serif italic text-[#0F172A] leading-tight animate-fade-in">
                &quot;{botMessage}&quot;
              </p>
            ) : (
              <p className="text-[#0F172A]/15 italic text-2xl font-serif">
                Analyze your tone...
              </p>
            )}
          </div>
        </div>
      </div>

      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;400;600&family=Playfair+Display:italic,wght@0,400..900;1,400..900&display=swap');
        body { font-family: 'Outfit', sans-serif; background-color: #FAFAFA; }
        .font-serif { font-family: 'Playfair Display', serif; }
        .custom-scrollbar::-webkit-scrollbar { width: 3px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #D4AF3744; border-radius: 10px; }
        @keyframes fade-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fade-in { animation: fade-in 0.8s ease-out forwards; }
      `}</style>
    </div>
  );
}
