import React, { Component, Fragment } from 'react';
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
  width: 85%;
  border: solid 1px;
  margin-right: 2vw;
  padding: 1%;
  border-radius: 5px;
  box-shadow: 0 15px 20px 2px #444;
  background-color: white;
`

const StyledBrowser = styled.div`
  display: flex;
  bottom: -7px;
  position: absolute;
  z-index: 2;
  flex-direction: row;
  justify-content: flex-end;
  overflow-x: auto;
  overflow-y: hidden;
  max-height: 20vh;
  width: 100vw;
`

const StyledResult = styled.span`
  cursor: pointer;
  padding: 0 2% 0 0;
  min-width: 7vw;
  opacity: 0.5;
  margin: auto;
  :hover {
    text-decoration: underline;
  }
  :nth-child(2) {
    margin-left: 3vw;
  }
`

const CloseX = styled.span`
  font-weight: 700;
  position: fixed;
  font-size: 1.5rem;
  cursor: pointer;
  z-index: 1;
  opacity: 0.8;
  :hover {
    opacity: 1;
  }
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

  clearSearch = () => {
    this.setState({ search: '' })
  }

  handleChange = (e) => {
    const { value } = e.target;
    this.setState({ search: value.toLowerCase() });
  };

  render() {
    const { goToMoment, showSearch } = this.props;
    const { search } = this.state;
    if (!this.currentTranscript) return 'Transcript coming soon...';
    return showSearch && (
      <Fragment>
        <Search
          search={search}
          handleChange={this.handleChange}
          transcript={this.getTranscript}
        />
        <StyledBrowser>
          {search && <SearchResults>
            <CloseX role="button" onClick={this.clearSearch}>X</CloseX>
              {this.searchResults.length ?  this.searchResults.map((result) => {
                const onClick = () => goToMoment(result.moment);
                return (
                  <StyledResult
                    key={result.moment}
                    onClick={onClick}
                    onKeyUp={onClick}
                    role="button"
                    tabIndex="0"
                  >
                    {result.sentence}
                  </StyledResult>
                );
              }) : 'No results found!'}
            </SearchResults>
          }
        </StyledBrowser>
      </Fragment>
    )
  }
}

TranscriptBrowser.propTypes = {
  goToMoment: PropTypes.func.isRequired,
  lesson: PropTypes.number.isRequired,
};

export default TranscriptBrowser;
