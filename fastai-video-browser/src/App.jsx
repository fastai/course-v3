import React, { useState, useRef, useEffect, useCallback } from 'react';
import qs from 'query-string';

import TranscriptBrowser from './components/TranscriptBrowser';
import TabbedViews from './components/Layout/TabbedViews';
import VideoPlayer from './components/VideoPlayer';
import LessonsNav from './components/LessonsNav';
import Layout from './components/Layout';
import Notes from './components/Notes';
import { timestampToSeconds } from './utils/time'
import './App.css';


const App = () => {
  const [selectedLesson, setSelectedLesson] = useState(1);
  const [startAt, setStartAt] = useState(null);
  const playerRef = useRef(null);
  
  useEffect(() => {
    const parsed = qs.parse(window.location.search);
    setSelectedLesson(parseInt(parsed.lesson || 1, 10));
    setStartAt(parseInt(parsed.t, 10));
  }, [window.location.search]);
  
  const goto = useCallback((timestamp) => {
    const time = timestampToSeconds(timestamp);
    playerRef.current.seekTo(time)
    playerRef.current.getInternalPlayer().playVideo()
  }, [playerRef])

  const selectedPart = selectedLesson < 8 ? 0 : 1;

  return (
    <Layout
      LeftPanelContent={
        <LessonsNav lesson={selectedLesson} part={selectedPart} />
      }
      RightPanelContent={
        <TabbedViews>
          <Notes lesson={selectedLesson} />
          <TranscriptBrowser lesson={selectedLesson} goToMoment={goto}  />
        </TabbedViews>
      }>
        <VideoPlayer ref={playerRef} lesson={selectedLesson} startAt={startAt} />
    </Layout>
  );
}

export default App;
