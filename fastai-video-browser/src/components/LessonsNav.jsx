import React from 'react';
import styled, { css } from 'styled-components';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { FiExternalLink } from 'react-icons/fi';

import logo from '../images/logo.png'
import { LESSONS_NAMES, quickLinks } from '../data';
import { standard } from '../utils/easing';

const StyledLesson = styled(Link)`
  height: 3rem;
  cursor: pointer;
  width: 80%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 6px 0;
  padding: 2px 12px;
  border-radius: 7px;
  transition: all 0.2s ${standard};
  text-decoration: none;
  &:hover {
    background: rgba(52, 125, 190, 0.15);
  }
  
  ${props => props.selected && css`
    font-weight: bold;
    color: var(--text-light);
    &, &:hover {
      background: linear-gradient(90deg, #347DBE, #2FB4D6);
    }
  `}
`

const Header = styled.header`
  text-align: center;
  margin: 12px 0;
`;

const StyledLessons = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const QuickLinks = styled.div`
  background-color: #fafbfc;
  padding: 20px 12px;
  h2 {
    margin: 8px 0;
  }
`;

const LessonsList = ({ selectedLesson, selectedPart }) => {
  var partLessons = LESSONS_NAMES[selectedPart]
  return (
    <StyledLessons>
      {Object.keys(partLessons).map((i) => {
        const lesson = partLessons[i];
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

const LessonsPanel = ({ lesson, part }) => (
  <div style={{ display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'space-between' }}>
    <div>
      <Header>
        <a href="/" target="_blank" rel="noopener noreferrer">
          <img src={logo} alt="fast.ai" />
        </a>
      </Header>
      <LessonsList selectedLesson={lesson} selectedPart={part} />
    </div>
    <QuickLinks>
      <h2>Quick Links</h2>
        {quickLinks.map(link => (
          <a key={link.href} href={link.href} target="_blank" rel="noopener noreferrer">
            <p>{link.title} <FiExternalLink /></p>
          </a>
        ))}
    </QuickLinks>
  </div>
)

LessonsPanel.propTypes = {
  lesson: PropTypes.number.isRequired,
  part: PropTypes.oneOf([0, 1]).isRequired
}

export default LessonsPanel
