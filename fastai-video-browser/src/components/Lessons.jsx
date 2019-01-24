import React from 'react';
import styled, { css } from 'styled-components';
import { Link } from 'react-router-dom';

const LESSONS = {
  1: 'Lesson 1',
  2: 'Lesson 2',
  3: 'Lesson 3',
  4: 'Lesson 4',
  5: 'Lesson 5',
  6: 'Lesson 6',
  7: 'Lesson 7',
}

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

const Lessons = ({ selectedLesson }) => {
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

export default Lessons
