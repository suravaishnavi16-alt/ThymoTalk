"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Activity } from "lucide-react";

export default function Home() {
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [statusText, setStatusText] = useState("Ready to converse.");
  const [botMessage, setBotMessage] = useState("");
  const [latencyMs, setLatencyMs] = useState<number>(0);
  
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

  // Function to play audio from base64 string
  const playAudioBase64 = async (base64Data: string, isUrgent: boolean) => {
    if (!base64Data) return;
    
    const checkSilenceAndPlay = async () => {
      if (isPlayingRef.current) {
        if (isUrgent && currentAudioRef.current) {
          // Urgent emotional support can interrupt any ongoing playback.
          currentAudioRef.current.pause();
          currentAudioRef.current = null;
          isPlayingRef.current = false;
        } else {
          return;
        }
      }

      // If the message is urgent (Fear/Sad), we play immediately
      // If not, we wait for the user to stop talking (VAD)
      if (!isUrgent && analyserRef.current) {
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        let silenceCount = 0;
        
        // Polling loop to wait for silence
        while (silenceCount < 10) {
          analyserRef.current.getByteFrequencyData(dataArray);
          const averageVolume = dataArray.reduce((p, c) => p + c, 0) / dataArray.length;
          
          if (averageVolume < 10) silenceCount++;
          else silenceCount = 0;
          
          await new Promise(resolve => setTimeout(resolve, 100)); // check every 100ms
        }
      }

      isPlayingRef.current = true;
      const audioSrc = `data:audio/mp3;base64,${base64Data}`;
      const audio = new Audio(audioSrc);
      currentAudioRef.current = audio;
      audio.onended = () => {
        isPlayingRef.current = false;
        currentAudioRef.current = null;
      };
      audio.play().catch(e => {
        console.error("Audio playback error:", e);
        isPlayingRef.current = false;
        currentAudioRef.current = null;
      });
    };

    checkSilenceAndPlay();
  };

  const speakTextFallback = (text: string, isUrgent: boolean) => {
    if (!text || typeof window === "undefined" || !window.speechSynthesis) return;

    const speak = () => {
      if (isPlayingRef.current && !isUrgent) return;
      if (isUrgent) {
        window.speechSynthesis.cancel();
      }

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1;
      utterance.pitch = 1;
      utterance.onstart = () => {
        isPlayingRef.current = true;
      };
      utterance.onend = () => {
        isPlayingRef.current = false;
      };
      utterance.onerror = () => {
        isPlayingRef.current = false;
      };
      window.speechSynthesis.speak(utterance);
    };

    if (!isUrgent && analyserRef.current) {
      const poll = () => {
        if (!analyserRef.current || !isSessionActive) return;
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        const averageVolume = dataArray.reduce((p, c) => p + c, 0) / dataArray.length;
        if (averageVolume < 10) {
          speak();
        } else {
          setTimeout(poll, 100);
        }
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
      const response = await fetch("http://localhost:8000/analyze_chunk", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) return;

      const data = await response.json();
      if (requestId < latestAppliedRequestIdRef.current) return;
      latestAppliedRequestIdRef.current = requestId;
      setLatencyMs(Math.round(performance.now() - requestStart));

      if (data.emotion) {
        console.log("Backend analysis:", data);
        if (data.emotion !== "Neutral") {
          setStatusText(`Feeling: ${data.emotion}`);
        }
      }

      // For non-urgent responses, playback helper waits for silence before speaking.
      if (data.interrupt && data.message) {
        setBotMessage(data.message);
        if (data.audio) {
          playAudioBase64(data.audio, data.urgent || false);
        } else {
          speakTextFallback(data.message, data.urgent || false);
        }
      }
    } catch (err) {
      console.error("Chunk processing error:", err);
      // setStatusText("Connection error.");
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
      
      // Setup Analyser for Silence Detection
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      let isVoiceActive = false;
      let silenceStart = Date.now();
      let lastChunkStart = Date.now(); // forced chunk timer
      const SILENCE_THRESHOLD = 5; 
      const SILENCE_DURATION = 500;

      const captureLoop = () => {
        if (!analyserRef.current || !intervalRef.current) return;

        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        const averageVolume = dataArray.reduce((p, c) => p + c, 0) / dataArray.length;

        const isNowActive = averageVolume > SILENCE_THRESHOLD;
        
        if (isNowActive) {
          isVoiceActive = true;
          silenceStart = Date.now();
        }

        // 1. SILENCE TRIGGER: User stopped speaking
        const isSilentLongEnough = isVoiceActive && (Date.now() - silenceStart > SILENCE_DURATION);
        
        // 2. FORCED CHUNK TRIGGER: keep urgency checks within ~2s windows.
        const isTakingTooLong = Date.now() - lastChunkStart > 2000;

        if ((isSilentLongEnough || isTakingTooLong) && mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
            console.log(isSilentLongEnough ? "VAD: Silence detected" : "VAD: Forced chunk for speed");
            isVoiceActive = false;
            lastChunkStart = Date.now();
            lastStopReasonRef.current = isSilentLongEnough ? "silence" : "forced";
            mediaRecorderRef.current.stop();
        }
        
        requestAnimationFrame(captureLoop);
      };

      const createRecorder = () => {
        const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            processAudioChunk(event.data, lastStopReasonRef.current);
          }
        };
        // Automatically restart after stopping to catch next sentence
        recorder.onstop = () => {
          if (intervalRef.current) {
            const next = createRecorder();
            mediaRecorderRef.current = next;
            next.start();
          }
        };
        return recorder;
      };

      const initialRecorder = createRecorder();
      mediaRecorderRef.current = initialRecorder;
      initialRecorder.start();

      // We use intervalRef as a flag to keep the loop running
      intervalRef.current = setInterval(() => {}, 1000); 
      requestAnimationFrame(captureLoop);

      setIsSessionActive(true);
      setStatusText("Listening...");
      setBotMessage("");
    } catch (err) {
      console.error(err);
      setStatusText("Mic access error.");
    }
  };

  const stopSession = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (mediaRecorderRef.current) {
      if (mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      mediaRecorderRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
      analyserRef.current = null;
    }
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    isPlayingRef.current = false;
    setIsSessionActive(false);
    setStatusText("Session ended.");
    window.speechSynthesis.cancel();
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-[#FAFAFA] text-[#0F172A] selection:bg-[#D4AF37] selection:text-[#FAFAFA] transition-colors duration-500">
      <div className="max-w-3xl w-full flex flex-col items-center space-y-16">
        
        {/* Classy Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-light tracking-wide">
            THYMOTALK<span className="font-semibold text-[#D4AF37]">.AI</span>
          </h1>
          <p className="text-[#0F172A]/60 font-light tracking-wider uppercase text-xs">
            Emotional Intelligence • Real-time Voice Detection
          </p>
        </div>

        {/* Central Interaction Area */}
        <div className="flex flex-col items-center justify-center space-y-10 w-full py-8">
          <button
            onClick={isSessionActive ? stopSession : startSession}
            className={`relative flex items-center justify-center w-32 h-32 rounded-full transition-all duration-700 ease-in-out outline-none ${
              isSessionActive
              ? "bg-[#D4AF37] text-[#FAFAFA] shadow-[0_0_50px_rgba(212,175,55,0.3)] scale-110"
              : "bg-transparent border border-[#0F172A]/10 text-[#0F172A] hover:border-[#D4AF37] hover:text-[#D4AF37]"
            }`}
          >
            {isSessionActive ? <Activity className="w-12 h-12 animate-pulse" /> : <Mic className="w-12 h-12" strokeWidth={1.2} />}
          </button>

          <div className="text-center h-8">
            <p className={`text-xs uppercase tracking-[0.3em] transition-all duration-500 ${isSessionActive ? "text-[#D4AF37] font-medium" : "text-[#0F172A]/30"}`}>
              {statusText}
            </p>
          </div>
        </div>

        {/* Bot Message Area */}
        <div className="min-h-[160px] w-full text-center px-8 flex items-center justify-center bg-white/50 rounded-2xl backdrop-blur-sm border border-black/[0.03]">
          {botMessage ? (
            <p className="text-2xl md:text-3xl font-serif italic text-[#0F172A] leading-relaxed transition-all duration-700">
              &quot;{botMessage}&quot;
            </p>
          ) : (
            <p className="text-[#0F172A]/20 italic text-xl font-serif tracking-wide">
              {isSessionActive ? "I'm listening to the tone of your voice..." : "Start a session to begin"}
            </p>
          )}
        </div>

        {/* Heartbeat Status */}
        <div className="text-[10px] text-[#0F172A]/20 font-mono mt-12 flex items-center space-x-3">
           <div className={`w-2 h-2 rounded-full ${isSessionActive ? "bg-green-500 animate-pulse" : "bg-gray-300"}`}></div>
           <span>SYSTEM: {isSessionActive ? "ACTIVE" : "IDLE"}</span>
           <span>|</span>
           <span>LATENCY: {latencyMs}ms</span>
        </div>
      </div>

      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;400;600&family=Playfair+Display:italic,wght@0,400..900;1,400..900&display=swap');
        
        :root {
          --background: #FAFAFA;
          --foreground: #0F172A;
        }

        body {
          font-family: 'Outfit', sans-serif;
          background-color: var(--background);
          color: var(--foreground);
        }

        .font-serif {
          font-family: 'Playfair Display', serif;
        }

        @keyframes pulse-ring {
          0% { transform: scale(.33); }
          80%, 100% { opacity: 0; }
        }
      `}</style>
    </div>
  );
}
