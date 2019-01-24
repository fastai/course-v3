import React, { Fragment } from 'react';
import styled, { css } from 'styled-components';
import FontAwesome from 'react-fontawesome';
import { Link } from 'react-router-dom';
import Toggler from './Toggler';

const LESSONS = {
  1: 'Lesson 1',
  2: 'Lesson 2',
  3: 'Lesson 3',
  4: 'Lesson 4',
  5: 'Lesson 5',
  6: 'Lesson 6',
  7: 'Lesson 7',
}

const StyledPanel = styled.section`
  background-color: var(--fastai-blue);
  display: flex;
  z-index: 1;
  position: relative;
  text-align: right;
  flex-direction: column;
  flex: 2;
  width: 10rem;
  max-width: 10rem;
  padding: 0 1%;
  box-shadow: 2px 0 30px #444;
  border-right: 1px solid black;

  ${props => props.closed && css`
    background-color: var(--fastai-blue);
    display: flex;
    flex-direction: column;
    flex: 0;
    padding: 0;
  `}
`

const StyledLesson = styled(Link)`
  height: 3rem;
  width: 80%;
  text-align: center;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  border: solid 1px white;
  ${props => props.selected && css`
    padding: 0.9rem 0;
    font-weight: 700;
    border: solid 2px white;
  `}
`

const StyledLessons = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`

const LessonsList = ({ selectedLesson }) => {
  return (
    <StyledLessons>
      {Object.keys(LESSONS).map((i) => {
        const lesson = LESSONS[i];
        return (
          <Lesson selectedLesson={selectedLesson} lesson={lesson} num={i} key={lesson} />
        );
      })}
    </StyledLessons>
  )
}

const Lesson = ({ num, lesson, selectedLesson }) => (
  <StyledLesson
    key={`lesson-${num}`} // eslint-disable-line react/no-array-index-key
    role="button"
    tabIndex="0"
    selected={parseInt(num) === selectedLesson}
    to={`?lesson=${num}`}
  >
    {lesson}
  </StyledLesson>
)

const LessonsPanel = ({ showLessons, toggleLessons, lesson }) => (
  <StyledPanel closed={!showLessons}>
    <Toggler
      styles={{
        right: '-30px',
        border: {
          right: '1px solid black',
          top: '1px solid black',
          bottom: '1px solid black',
        }
      }}
      condition={showLessons}
      onClick={toggleLessons}
      iconTrue="fa-chevron-left"
      iconFalse="fa-chevron-right"
    />
      {showLessons && <Fragment>
          <header>
            <h1 style={{
              fontSize: '1.125rem',
              textAlign: 'center',
              fontFamily: 'Helvetica',
              color: 'white'
            }}>
              <FontAwesome className="fa-home" name="fa-home" />
              <a
                href="/"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  textDecoration: 'none',
                  marginLeft: '0.5rem'
                }}
              >course</a>
            </h1>
          </header>
          <LessonsList selectedLesson={lesson} />
        </Fragment>}
    </StyledPanel>
)

export default LessonsPanel
