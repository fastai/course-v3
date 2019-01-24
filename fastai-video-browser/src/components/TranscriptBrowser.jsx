import React, { Component } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import Search from './Search';
import lesson1Trans from '../assets/dl-1-1/transcript.json';
import lesson2Trans from '../assets/dl-1-2/transcript.json';
import lesson3Trans from '../assets/dl-1-3/transcript.json';
// import lesson4Trans from '../assets/dl-1-4/transcript.json'; coming soon!
import lesson5Trans from '../assets/dl-1-5/transcript.json';
import lesson6Trans from '../assets/dl-1-6/transcript.json';
// import lesson7Trans from '../assets/dl-1-7/transcript.json'; coming soon!


const TRANSCRIPTS = {
  1: lesson1Trans,
  2: lesson2Trans,
  3: lesson3Trans,
  4: null,
  5: lesson5Trans,
  6: lesson6Trans,
  7: null,
};

const SearchResults = styled.div`
  display: flex;
  flex-direction: row;
  overflow-x: auto;
  overflow-y: hidden;
  width: 100%;
`

class TranscriptBrowser extends Component {
  state = {
    search: '',
    currentMoment: null,
  };

  static getDerivedStateFromProps = (props, state) => {
    const transcript = TRANSCRIPTS[props.lesson];
    if (!transcript) return { ...state };
    return { ...state };
  };

  get currentTranscript() {
    const { lesson } = this.props;
    return TRANSCRIPTS[lesson];
  }

  get searchResults() {
    const transcript = this.currentTranscript;
    const { search } = this.state;
    if (!transcript) return []
    return Object.keys(transcript)
      .filter((timestamp) =>
        transcript[timestamp].toLowerCase().includes(search),
      )
      .map((timestamp) => ({
        moment: timestamp,
        sentence: transcript[timestamp],
      }))
      .slice(0, 12);
  }

  handleChange = (e) => {
    const { value } = e.target;
    this.setState({ search: value.toLowerCase() });
  };

  render() {
    const { goToMoment } = this.props;
    const { search } = this.state;
    if (!this.currentTranscript) return 'Transcript coming soon...';
    return (
      <div className="TranscriptBrowser">
        <div className="top">
          <Search
            search={search}
            handleChange={this.handleChange}
            transcript={this.getTranscript}
          />
          <SearchResults>
            {search && this.searchResults.map((result) => {
              const onClick = () => goToMoment(result.moment);
              return (
                <span
                  key={result.moment}
                  onClick={onClick}
                  onKeyUp={onClick}
                  role="button"
                  tabIndex="0"
                  className="search-result"
                >
                  {result.sentence}
                </span>
              );
            })}
          </SearchResults>
        </div>
      </div>
    );
  }
}

TranscriptBrowser.propTypes = {
  goToMoment: PropTypes.func.isRequired,
  lesson: PropTypes.number.isRequired,
};

export default TranscriptBrowser;
