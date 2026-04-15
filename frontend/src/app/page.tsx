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
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";
      const response = await fetch(`${apiUrl}/analyze_chunk`, {
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
      const SILENCE_DURATION = 300;

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
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-[#0B0F1A] text-white selection:bg-[#A855F7]/30">
      <div className="max-w-4xl w-full flex flex-col items-center space-y-12 z-10">

        {/* Header (Premium AI Dark Mode) */}
        <div className="text-center space-y-3">
          <h1 className="text-4xl md:text-6xl font-light tracking-wide bg-clip-text text-transparent bg-gradient-to-r from-[#A855F7] via-[#F472B6] to-[#22D3EE]">
            THYMOTALK<span className="font-semibold text-white">.AI</span>
          </h1>
          <p className="text-white/40 uppercase text-[10px] md:text-xs tracking-[0.25em] font-medium pulse-glow">
            Intelligence • Analysis • Transcription
          </p>
        </div>

        {/* Action & Stats Row (UI Modernized) */}
        <div className="flex flex-col md:flex-row items-center justify-center gap-12 w-full">
          <div className="flex flex-col items-center space-y-6">
            <button
              onClick={isSessionActive ? stopSession : startSession}
              className={`relative flex items-center justify-center w-36 h-36 rounded-full transition-all duration-300 ease-in-out border md:hover:scale-[1.03] shadow-lg
                ${isSessionActive 
                  ? "bg-white/5 border-[#A855F7]/50 shadow-[#A855F7]/20 glow-active scale-105" 
                  : "bg-white/5 border-white/10 hover:border-[#A855F7]/40 backdrop-blur-md"
                }`}
            >
              {/* Outer rings for visual depth */}
              {isSessionActive && (
                <>
                  <div className="absolute inset-0 rounded-full border border-[#F472B6]/30 animate-pulse-slow scale-[1.15]" />
                  <div className="absolute inset-0 rounded-full border border-[#22D3EE]/20 animate-pulse-slow scale-[1.3] animation-delay-200" />
                </>
              )}
              {isSessionActive ? (
                <Activity className="w-14 h-14 text-[#22D3EE] animate-pulse drop-shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
              ) : (
                <Mic className="w-14 h-14 text-white/70" strokeWidth={1} />
              )}
            </button>
            <div className="flex flex-col items-center space-y-1">
              <span className={`text-[10px] font-bold uppercase tracking-widest transition-colors ${isSessionActive ? "text-[#22D3EE] drop-shadow-[0_0_2px_rgba(34,211,238,0.8)]" : "text-white/40"}`}>
                {isSessionActive ? "ACTIVE" : "IDLE"}
              </span>
              <span className="text-[9px] text-white/20 font-mono">LATENCY: {latencyMs}ms</span>
            </div>
          </div>

          <div className="flex-1 w-full md:max-w-md bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 shadow-xl min-h-[144px] flex flex-col transition-all duration-500">
            <div className="flex items-center space-x-2 text-[#A855F7] mb-3">
              <MessageSquare className="w-4 h-4" />
              <span className="text-[10px] font-bold uppercase tracking-tighter text-white/80">Real-time Transcript</span>
            </div>
            <div className="flex-1 overflow-y-auto max-h-[120px] pr-2 custom-scrollbar">
              {fullTranscript.length > 0 ? (
                <div className="space-y-2">
                  {fullTranscript.map((line, idx) => (
                    <p key={idx} className={`text-sm leading-relaxed transition-opacity duration-300 ${idx === fullTranscript.length - 1 ? "text-white font-normal" : "text-white/30"}`}>
                      {line}
                      {idx === fullTranscript.length - 1 && isSessionActive && (
                        <span className="inline-block w-1 h-3 ml-1 bg-[#22D3EE] animate-blink" />
                      )}
                    </p>
                  ))}
                </div>
              ) : (
                <p className="text-white/20 text-sm italic">
                  {isSessionActive ? "Waiting for speech" : "Transcript will appear here"}
                  {isSessionActive && <span className="animate-dots"></span>}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Emotion Display Area (Premium Animation) */}
        <div className="w-full text-center space-y-6 max-w-2xl">
          <div className="h-6">
            <span className="text-[10px] text-[#A855F7] font-bold uppercase tracking-[0.3em] drop-shadow-[0_0_2px_rgba(168,85,247,0.5)]">
              {statusText}
            </span>
          </div>
          <div className="min-h-[120px] flex items-center justify-center p-8 bg-white/5 backdrop-blur-xl rounded-3xl border border-white/5 shadow-2xl relative overflow-hidden transition-all duration-500">
            {/* Subtle background glow based on active state */}
            {isSessionActive && <div className="absolute inset-0 bg-gradient-to-r from-[#A855F7]/10 via-transparent to-[#22D3EE]/10 opacity-50 blur-3xl animate-pulse-slow" />}
            
            <div className="relative z-10 w-full flex justify-center">
              {botMessage ? (
                botMessage.includes("Thinking") || botMessage === "Listening..." ? (
                  <p className="text-2xl md:text-3xl font-serif italic text-white/60 leading-tight">
                    {botMessage}
                  </p>
                ) : (
                  <p className="text-3xl md:text-4xl font-serif italic text-white leading-tight animate-slide-up-fade">
                    &quot;{botMessage}&quot;
                  </p>
                )
              ) : (
                <p className="text-white/20 italic text-2xl font-serif">
                  Analyze your tone...
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Decorative Background Elements */}
      <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-[#A855F7]/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-[#22D3EE]/10 rounded-full blur-[120px] pointer-events-none" />

      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;400;600&family=Playfair+Display:italic,wght@0,400..900;1,400..900&display=swap');
        body { font-family: 'Outfit', sans-serif; background-color: #0B0F1A; }
        .font-serif { font-family: 'Playfair Display', serif; }
        
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(168,85,247,0.3); border-radius: 10px; }
        
        .animation-delay-200 { animation-delay: 200ms; }
        
        .glow-active {
          box-shadow: 0 0 30px -5px rgba(168,85,247, 0.4), inset 0 0 15px -5px rgba(244,114,182, 0.3);
        }
        
        @keyframes slide-up-fade {
          0% { opacity: 0; transform: translateY(15px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .animate-slide-up-fade { animation: slide-up-fade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
        
        @keyframes pulse-slow {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.05); }
        }
        .animate-pulse-slow { animation: pulse-slow 3s ease-in-out infinite; }
        
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        .animate-blink { animation: blink 1s step-end infinite; }
        
        @keyframes dots {
          0%, 20% { color: rgba(255,255,255,0); text-shadow: .25em 0 0 rgba(255,255,255,0), .5em 0 0 rgba(255,255,255,0); }
          40% { color: white; text-shadow: .25em 0 0 rgba(255,255,255,0), .5em 0 0 rgba(255,255,255,0); }
          60% { text-shadow: .25em 0 0 white, .5em 0 0 rgba(255,255,255,0); }
          80%, 100% { text-shadow: .25em 0 0 white, .5em 0 0 white; }
        }
        .animate-dots::after {
          content: ' .';
          animation: dots 1.5s steps(5, end) infinite;
        }
      `}</style>
    </div>
  );
}
