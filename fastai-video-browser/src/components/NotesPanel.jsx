import React from 'react';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';
import CodeBlock from './CodeBlock';
import Toggler from './Toggler';
import lesson1Notes from '../assets/dl-1-1/notes.md';
import lesson2Notes from '../assets/dl-1-2/notes.md';
import lesson3Notes from '../assets/dl-1-3/notes.md';
import lesson4Notes from '../assets/dl-1-4/notes.md';
import lesson5Notes from '../assets/dl-1-5/notes.md';
import lesson6Notes from '../assets/dl-1-6/notes.md';
import lesson7Notes from '../assets/dl-1-7/notes.md';

const NOTES = {
  1: lesson1Notes,
  2: lesson2Notes,
  3: lesson3Notes,
  4: lesson4Notes,
  5: lesson5Notes,
  6: lesson6Notes,
  7: lesson7Notes,
}

const StyledPanel = styled.section`
  height: 100vh;
  display: flex;
  flex-direction: column;
  position: relative;
  border-left: solid 1px var(--fastai-blue);
  flex: ${props => props.open ? '3' : 0};
  max-width: 35vw;
  background-color: white;
`

const StyledMarkdown = styled(ReactMarkdown)`
  padding: 0 2%;
  overflow-y: scroll;
`

const CACHE = {}

class MarkdownRenderer extends React.Component {
  state = {
    notes: '',
    rendered: null,
  }

  componentDidMount() {
    this.fetchLesson()
  }

  componentDidUpdate() {
    if (this.props.lesson !== this.state.rendered) this.fetchLesson()
  }

  fetchLesson() {
    const cachedNotes = CACHE[this.props.lesson]
    if (cachedNotes) return this.setState({ notes: cachedNotes, rendered: this.props.lesson })
    fetch(NOTES[this.props.lesson])
      .then(res => res.text())
      .then(rawMd => CACHE[this.props.lesson] = rawMd)
      .then(notes => this.setState({ notes, rendered: this.props.lesson }))
      .catch(console.error)
  }

  render() {
    return <StyledMarkdown source={this.state.notes} renderers={{ code: CodeBlock }} />
  }
}

const NotesPanel = ({ lesson, showNotes, toggleNotes, ...rest }) => (
  <StyledPanel open={showNotes} {...rest}>
    <Toggler
      styles={{ left: "-30px" }}
      condition={showNotes}
      onClick={toggleNotes}
      iconTrue="fa-chevron-right"
      iconFalse="fa-chevron-left"
    />
    {showNotes && <MarkdownRenderer lesson={lesson} />}
  </StyledPanel>
)

export default NotesPanel
