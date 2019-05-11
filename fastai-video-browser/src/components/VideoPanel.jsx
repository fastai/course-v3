import React from 'react';
import styled from 'styled-components';
import VideoPlayer from './VideoPlayer';

const StyledPanel = styled.div`
  display: flex;
  flex-direction: column;
  flex: 6;
  background-color: white;
`

const VideoPanel = React.forwardRef(({ lesson, startAt }, ref) => (
  <StyledPanel>
    <VideoPlayer
      lesson={lesson}
      ref={ref}
      startAt={startAt}
    />
  </StyledPanel>
))

export default VideoPanel
