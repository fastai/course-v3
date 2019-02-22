import React, { Component, Fragment } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import Search from './Search';

const TRANSCRIPTS = {
  1: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/transcripts/transcript-1-1.json',
  2: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/transcripts/transcript-1-2.json',
  3: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/transcripts/transcript-1-3.json',
  4: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/transcripts/transcript-1-4.json',
  5: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/transcripts/transcript-1-5.json',
  6: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/transcripts/transcript-1-6.json',
  7: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/transcripts/transcript-1-7.json',
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
  bottom: 0px;
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
    const toFetch = TRANSCRIPTS[this.props.lesson]
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
            {result.sentence}
          </StyledResult>
        );
      })
    }
    return 'No results found.'
  }

  render() {
    const { showSearch } = this.props;
    const { search, transcript } = this.state;
    return showSearch && (
      <Fragment>
        <Search
          search={search}
          handleChange={this.handleChange}
          transcript={this.getTranscript}
        />
        <StyledBrowser>
          {search && <SearchResults>
              {transcript && <CloseX role="button" onClick={this.clearSearch}>X</CloseX>}
                {this.results}
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
