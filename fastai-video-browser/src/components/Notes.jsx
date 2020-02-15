import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';
import PropTypes from 'prop-types';

import CodeBlock from './CodeBlock';
import { NOTES_URLS } from '../data';

const StyledMarkdown = styled(ReactMarkdown)`
  padding: 12px 24px;
  p {
    line-height: 1.5em;
    margin-bottom: 24px;
  }

  h1, h2, h3, h4, h5, h6 {
    margin-bottom: 24px;
    font-weight: bold;
  }
`;

const CACHE = {}

const fetchLesson = async (id) => {
  const cache = CACHE[id];
  if (cache) return cache;

  const res = await fetch(NOTES_URLS[id]);
  const text = await res.text();
  CACHE[id] = text;
  return text;
}

const Notes = ({ lesson }) => {
  const [notes, setNotes] = useState('');

  useEffect(() => { 
    fetchLesson(lesson)
    .then(text => setNotes(text))
    .catch(err => console.error(err))
  }, [lesson]);

  return (
    <StyledMarkdown linkTarget="_blank" source={notes} renderers={{ code: CodeBlock }} />
  )
}

Notes.propTypes = {
  lesson: PropTypes.number.isRequired
}

export default Notes
