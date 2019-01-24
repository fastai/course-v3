/* global location */
import React, { Component, Fragment } from 'react';
import styled from 'styled-components';
import FontAwesome from 'react-fontawesome';
import VideoPlayer from './components/VideoPlayer';
import Toggler from './components/Toggler';
import Lessons from './components/Lessons';
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
    return { ...state, selectedLesson: parseInt(parsed.lesson) || 1 }
  }

  state = {
    showLessons: true,
    showNotes: false,
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
    const { toggleLessons, toggleNotes } = this;
    const { showLessons, showNotes, selectedLesson } = this.state;
    return (
        <StyledApp>
          <section className={`left ${showLessons ? '' : 'closed'}`}>
            <Toggler
              styles={{ right: '-30px' }}
              condition={showLessons}
              onClick={toggleLessons}
              iconTrue="fa-chevron-left"
              iconFalse="fa-chevron-right"
            />
            {showLessons && (
              <Fragment>
                <header>
                <h1 style={{ fontSize: '1.125rem', textAlign: 'center', fontFamily: 'Helvetica', color: 'white' }}>
                    <FontAwesome className="fa-home" name="fa-home" />
                    <a
                      href="/"
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ textDecoration: 'none', marginLeft: '0.5rem' }}
                    >
                      course
                    </a>
                  </h1>
                </header>
                <Lessons selectedLesson={selectedLesson} />
              </Fragment>
            )}
          </section>
          <section className="center">
            <div className="row">
              <VideoPlayer lesson={selectedLesson} ref={this.videoPlayer} />
            </div>
            <TranscriptBrowser
              lesson={selectedLesson}
              goToMoment={this.goToMoment}
            />
          </section>
          <NotesPanel lesson={selectedLesson} toggleNotes={toggleNotes} showNotes={showNotes} />
        </StyledApp>
    );
  }
}

export default App;
