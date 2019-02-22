import React from 'react';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';
import CodeBlock from './CodeBlock';
import Toggler from './Toggler';

const NOTES = {
  1: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/notes/notes-1-1.md',
  2: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/notes/notes-1-2.md',
  3: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/notes/notes-1-3.md',
  4: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/notes/notes-1-4.md',
  5: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/notes/notes-1-5.md',
  6: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/notes/notes-1-6.md',
  7: 'https://raw.githubusercontent.com/fastai/course-v3/master/files/dl-2019/notes/notes-1-7.md',
}

const StyledPanel = styled.section`
  height: 100vh;
  display: flex;
  z-index: 1;
  flex-direction: column;
  position: relative;
  border-left: solid 1px var(--fastai-blue);
  flex: ${props => props.open ? 3 : 0};
  max-width: 35vw;
  background-color: white;
  box-shadow: -1px 0 30px #444;
`

const StyledMarkdown = styled(ReactMarkdown)`
  padding: 0 2%;
  overflow-y: auto;
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
    /*
     * We `fetch` our own resource (a Webpack-resolved relative URL) so that React can parse the contents of
     * referenced markdown file without any fancy configuration in Webpack.
     */
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
      styles={{
        left: '-32px',
        zIndex: 1,
        border: {
          left: '1px solid black',
          bottom: '1px solid black',
          top: '1px solid black',
        }
      }}
      condition={showNotes}
      onClick={toggleNotes}
      iconTrue="fa-chevron-right"
      iconFalse="fa-chevron-left"
    />
    {showNotes && <MarkdownRenderer lesson={lesson} />}
  </StyledPanel>
)

export default NotesPanel
