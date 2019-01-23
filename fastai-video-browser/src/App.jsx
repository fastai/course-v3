/* global location */
import React, { Component, Fragment } from 'react';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';
import CodeBlock from './components/CodeBlock';
import FontAwesome from 'react-fontawesome';
import VideoPlayer from './components/VideoPlayer';
import Toggler from './components/Toggler';
import Lessons from './components/Lessons';
import TranscriptBrowser from './components/TranscriptBrowser';
import qs from 'query-string';
import './App.css';

const StyledApp = styled.div`
  height: 100vh;
  width: 100vw;
  display: flex;
  flex-direction: row;
  font-family: 'PT Sans', Helvetica, Arial, sans-serif;
`;

const CHAPTERS = null; // ['Chapter 1', 'Chapter 2', 'Chapter 3', 'Chapter 4', 'Chapter 5']

const getMinutes = (seconds) =>
  (seconds.toFixed(0) / 60).toString().split('.')[0];

const secondsToTimestamp = (totalSeconds) => {
  let minutes = getMinutes(totalSeconds);
  let remainder = (totalSeconds.toFixed(0) % 60).toString();
  if (minutes.length < 2) minutes = `0${minutes}`;
  if (remainder.length < 2) remainder = `0${remainder}`;
  return `${minutes}:${remainder}`;
};

const timestampToSeconds = (moment) => {
  const [minutes, seconds] = moment.split(':');
  return Number(minutes) * 60 + Number(seconds);
};

class App extends Component {
  constructor(props) {
    super(props);
    this.videoPlayer = React.createRef();
    this.currentMomentInterval = null;
  }

  static getDerivedStateFromProps(props, state) {
    const parsed = qs.parse(window.location.search)
    return { ...state, selectedLesson: parseInt(parsed.lesson) || 1 }
  }

  state = {
    showLessons: true,
    showNotes: false,
    selectedLesson: 1,
    currentMoment: '00:00',
  };

  componentDidMount() {
    this.pollForCurrentMoment();
  }

  pollForCurrentMoment = () => {
    const { selectedLesson } = this.state;

    if (this.currentMomentInterval) return;
    this.currentMomentInterval = setInterval(() => {
      const curTime = this.videoPlayer.current.getCurrentTime();
      if (!curTime) return;
      const timestamp = secondsToTimestamp(curTime);
      this.setState({ currentMoment: timestamp });
    }, 500);
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
    const { showLessons, showNotes, selectedLesson, currentMoment } = this.state;
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
                    <FontAwesome className="fa-home" />
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
              {CHAPTERS && (
                <div className="chapter-nav white">
                  {CHAPTERS.map((chap) => (
                    <div key={chap} className="chapter ba grow">
                      {chap}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <TranscriptBrowser
              lesson={selectedLesson}
              goToMoment={this.goToMoment}
              currentMoment={currentMoment}
            />
          </section>
          <NotesPanel toggleNotes={toggleNotes} showNotes={showNotes} />
        </StyledApp>
    );
  }
}

const StyledPanel = styled.section`
  position: relative;
  width: ${props => props.open ? '25vw' : '0'};;
  height: 100vh;
  display: flex;
  flex-direction: column;
  transition: width 0.4s ease-in-out;
`

const source = `## HTML block below

> This blockquote will change based on the HTML settings above.

## How about some code ?
\`\`\`js
var React = require('react');
var Markdown = require('react-markdown');
console.log('hello')

React.render(
  <Markdown source="# Your markdown here" />,
  document.getElementById('content')
);
\`\`\`
`


const NotesPanel = ({ showNotes, toggleNotes, ...rest }) => (
  <StyledPanel open={showNotes} {...rest}>
    <Toggler
      styles={{ left: '-30px' }}
      condition={showNotes}
      onClick={toggleNotes}
      iconTrue="fa-chevron-right"
      iconFalse="fa-chevron-left"
    />
    <ReactMarkdown source={source} renderers={{
      code: CodeBlock
    }} />
  </StyledPanel>
)

export default App;
