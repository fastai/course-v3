import React, { useState, useEffect, useCallback, useMemo } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import { TiArrowRight } from 'react-icons/ti';

import Search from './Search';
import { TRANSCRIPT_URLS } from '../data';
import { standard } from '../utils/easing';

const SearchResults = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
`;

const StyledBrowser = styled.div`
  width: 100%;
  height: 100%;
`

const StyledResult = styled.span`
  cursor: pointer;
  padding: 23px 18px;
  border-bottom: 1px solid #eee;
  display: flex;
  align-items: center;
  justify-content: space-between;

  div > p {
    margin: 4px 0;
  }
  
  div:first-child > span {
    opacity: 0.4;
  }

  div:last-child {
    opacity: 0;
    transition: all 0.4s ${standard};
    transform: translateX(-40px);
  }
  
  &:hover {
    background: linear-gradient(90deg, #347DBE, #2FB4D6);
    color: #fff;
    div:last-child {
      opacity: 1.0;
      transform: translateX(0px);
    }
  }
`;

const CACHE = {}

const fetchTranscript = async (lesson) => {
  const cache = CACHE[lesson];
  if (cache) return cache;
  
  const url = TRANSCRIPT_URLS[lesson]
  if (!url) return null;
  
  const res = await fetch(url)
  const text = await res.json();
  CACHE[lesson] = text;
  
  return text;
}

const TranscriptUnavailable = () => (
  <div style={{ display: 'flex', width: '100%', height: '100%', alignItems: 'center', justifyContent: 'center' }}>
    <p style={{ color: '#444' }}>The transcript for this lesson is currently unavailable</p>
  </div>
)

// Bolden the first occurence of the query in the search result 
const ResultText = ({ result }) => {
  const [start, length] = result.occurence;
  const end = start + length;
  
  const startSlice = result.sentence.slice(0, start);
  const occurence = result.sentence.slice(start, end);
  const endSlice = result.sentence.slice(end, result.sentence.length);
  
  return <p>{ startSlice }<b>{ occurence }</b>{ endSlice }</p>
 }

const Result = ({ result }) => (
  <StyledResult
    onClick={result.goto}
    onKeyUp={result.goto}
    role="button"
    tabIndex="0">
    <div>
      <ResultText result={result} />
      <span>{ result.moment }</span>
    </div>
    <div>Seek <TiArrowRight /></div>
  </StyledResult>
);

const TranscriptBrowser = ({ lesson, goToMoment }) => {
  const [search, setSearch] = useState('');
  const [transcript, setTranscript] = useState(null);
  
  useEffect(() => {
    fetchTranscript(lesson).then(text => {
      setTranscript(text);
      setSearch('')
    }).catch(err => console.error(err))
  }, [lesson]);

  const handleChange = useCallback(evt => setSearch(evt.target.value), []);
  
  const results = useMemo(() => {
    if (!transcript) return [];
    return Object.keys(transcript)
      .map(timestamp => ({
        timestamp,
        occurence: [transcript[timestamp].toLowerCase().indexOf(search), search.length]
      }))
      .filter(result => result.occurence[0] !== -1)
      .map(result => ({
        moment: result.timestamp,
        sentence: transcript[result.timestamp],
        goto: () => goToMoment(result.timestamp),
        occurence: result.occurence
      }))
      .slice(0, 12);
  }, [search, goToMoment, transcript]);

  return (
    <>
      <Search
        search={search}
        handleChange={handleChange}
        disabled={!transcript} />
      <StyledBrowser>
        { !transcript && <TranscriptUnavailable /> }
        { (transcript && search.length > 0) && (
          <SearchResults>
            { results.map(result => <Result key={result.moment} result={result} /> ) }
          </SearchResults>
        )}
      </StyledBrowser>
    </>
  )
}

TranscriptBrowser.propTypes = {
  goToMoment: PropTypes.func.isRequired,
  lesson: PropTypes.number.isRequired,
};

export default TranscriptBrowser;
