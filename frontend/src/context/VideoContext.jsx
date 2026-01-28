import { createContext, useContext, useState } from 'react';

const VideoContext = createContext();

export function VideoProvider({ children }) {
  const [currentVideoId, setCurrentVideoId] = useState(null);
  const [currentVideoName, setCurrentVideoName] = useState(null);
  const [chatSessionId, setChatSessionId] = useState(null);

  return (
    <VideoContext.Provider
      value={{
        currentVideoId,
        setCurrentVideoId,
        currentVideoName,
        setCurrentVideoName,
        chatSessionId,
        setChatSessionId,
      }}
    >
      {children}
    </VideoContext.Provider>
  );
}

export function useVideo() {
  return useContext(VideoContext);
}
