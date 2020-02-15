import React, { Component, Fragment } from 'react';
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

class TranscriptBrowser extends Component {
  state = {
    search: '',
    transcript: '',
    rendered: null,
  };

  componentDidMount() {
    this.fetchTranscript()
  }

  componentDidUpdate() {
    if (this.props.lesson !== this.state.rendered)this.fetchTranscript()
  }

  fetchTranscript() {
    const cached = CACHE[this.props.lesson]
    if (cached) return this.setState({ transcript: cached, rendered: this.props.lesson })
    /*
     * We `fetch` our own resource (a Webpack-resolved relative URL) so that React can parse the contents of
     * referenced markdown file without any fancy configuration in Webpack.
     */
    const toFetch = TRANSCRIPT_URLS[this.props.lesson]
    if (!toFetch) return this.setState({
      transcript: null,
      rendered: this.props.lesson,
    })
    fetch(toFetch)
      .then(res => res.json())
      .then(rawMd => CACHE[this.props.lesson] = rawMd)
      .then(transcript => this.setState({
        transcript,
        rendered: this.props.lesson
      }))
      .catch(console.error)
  }


  get searchResults() {
    const { search, transcript } = this.state;
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

  get results() {
    const { goToMoment } = this.props;
    const { transcript } = this.state;
    if (!transcript) return <span style={{ marginLeft: '25%' }}>Transcript coming soon...</span>
    if (this.searchResults.length) {
      return this.searchResults.map((result) => {
        const onClick = () => goToMoment(result.moment);
        return (
          <StyledResult
            key={result.moment}
            onClick={onClick}
            onKeyUp={onClick}
            role="button"
            tabIndex="0"
          >
            <div>
              <p>{ result.sentence }</p>
              <span>{ result.moment }</span>
            </div>
            <div>Seek <TiArrowRight /></div>
          </StyledResult>
        );
      })
    }
    return (
      'No results found.'
    )
  }

  render() {
    const { search, transcript } = this.state;
    return (
      <Fragment>
        <Search
          search={search}
          handleChange={this.handleChange}
          transcript={this.getTranscript}
        />
        <StyledBrowser>
          { transcript && <SearchResults>
            {this.results}
          </SearchResults> }
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
