/* global location */
import React, { Component, Fragment } from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
import FontAwesome from 'react-fontawesome';
import VideoPlayer from './components/VideoPlayer';
import TranscriptBrowser from './components/TranscriptBrowser';
import qs from 'query-string';
import './App.css';

const StyledToggleWrapper = styled.span`
  padding: 5% 0;
  z-index: 20;
  color: white;
  cursor: pointer;
  font-size: 2rem;
  width: 30px;
  position: absolute;
  top: 1.7rem;
  text-align: center;
  right: -30px;
  background-color: #202020;
`

const Icon = styled(FontAwesome)`
  vertical-align: middle;
  font-size: 1rem;
`

const Toggler = ({ onClick, condition, iconTrue, iconFalse }) => (
  <StyledToggleWrapper onClick={onClick} role="button" tabIndex="0">
    {condition ? <Icon size="1x" className={iconTrue} /> : <Icon className={iconFalse} size="1x" />}
  </StyledToggleWrapper>
)

const StyledApp = styled.div`
  height: 100vh;
  width: 100vw;
  display: flex;
  flex-direction: row;
  font-family: 'PT Sans', Helvetica, Arial, sans-serif;
`;

const LESSONS = {
  1: 'Lesson 1',
  2: 'Lesson 2',
  3: 'Lesson 3',
  4: 'Lesson 4',
  5: 'Lesson 5',
  6: 'Lesson 6',
  7: 'Lesson 7',
}

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

  selectLesson = (selectedLesson) => {
    this.setState({ selectedLesson });
  };

  render() {
    const { toggleLessons, selectLesson } = this;
    const { showLessons, selectedLesson, currentMoment } = this.state;
    return (
        <StyledApp>
          <section className={`left ${showLessons ? '' : 'closed'}`}>
            <Toggler
              condition={showLessons}
              onClick={toggleLessons}
              iconTrue="fa-chevron-left"
              iconFalse="fa-chevron-right"
            />
            {showLessons && (
              <Fragment>
                <header className="App-header serif">
                  <h1 className="f2 underline white tc">
                    <a
                      href="http://fast.ai"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      fast.ai
                    </a>
                  </h1>
                </header>
                <div className="lessons white">
                  {Object.keys(LESSONS).map((i) => {
                    const lesson = LESSONS[i];
                    return (
                        <Link
                          key={`lesson-${i}`} // eslint-disable-line react/no-array-index-key
                          role="button"
                          tabIndex="0"
                          className={`${
                            i === selectedLesson ? 'selected' : ''
                          } lesson ba ${
                            lesson === 'Coming Soon!' ? 'disabled' : 'grow'
                          }`}
                          to={`?lesson=${i}`}
                        >
                          {lesson}
                        </Link>
                    );
                  })}
                </div>
              </Fragment>
            )}
          </section>
          <section className="right">
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
        </StyledApp>
    );
  }
}

export default App;
