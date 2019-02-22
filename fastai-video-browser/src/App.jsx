import React, { Component } from 'react';
import styled from 'styled-components';
import VideoPanel from './components/VideoPanel';
import LessonsPanel from './components/LessonsPanel';
import NotesPanel from './components/NotesPanel';
import TranscriptBrowser from './components/TranscriptBrowser';
import { timestampToSeconds } from './utils/time'
import qs from 'query-string';
import './App.css';

const StyledApp = styled.div`
  height: 100vh;
  width: 100vw;
  display: flex;
  flex-direction: row;
  font-family: 'PT Sans', Helvetica, Arial, sans-serif;
`;

class App extends Component {
  constructor(props) {
    super(props);
    this.videoPlayer = React.createRef();
  }

  static getDerivedStateFromProps(props, state) {
    const parsed = qs.parse(window.location.search)
    // Parse selected lesson from query string, so we don't have to deal with real /routing.
    return { ...state, selectedLesson: parseInt(parsed.lesson) || 1 }
  }

  state = {
    // We show both panels by default to illustrate UX to first-time users.
    showSearch: true,
    showLessons: true,
    showNotes: true,
    selectedLesson: 1,
  };

  goToMoment = (timestamp) => {
    const time = timestampToSeconds(timestamp);
    this.videoPlayer.current.seekTo(time)
    this.videoPlayer.current.getInternalPlayer().playVideo()
  };

  toggleLessons = () => {
    const { showLessons } = this.state;
    this.setState({ showLessons: !showLessons });
  };

  toggleNotes = () => {
    const { showNotes } = this.state;
    this.setState({ showNotes: !showNotes })
  }

  render() {
    const {
      toggleLessons,
      toggleNotes,
    } = this;
    const {
      showLessons,
      showNotes,
      showSearch,
      selectedLesson,
    } = this.state;
    return (
        <StyledApp>
          <LessonsPanel
            showLessons={showLessons}
            toggleLessons={toggleLessons}
            lesson={selectedLesson}
          />
          <VideoPanel
            ref={this.videoPlayer}
            lesson={selectedLesson}
          />
          <NotesPanel
            showNotes={showNotes}
            toggleNotes={toggleNotes}
            lesson={selectedLesson}
          />
          <TranscriptBrowser
            showSearch={showSearch}
            goToMoment={this.goToMoment}
            lesson={selectedLesson}
          />
        </StyledApp>
    );
  }
}

export default App;
