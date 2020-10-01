import React, { forwardRef } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import YouTubePlayer from 'react-player/lib/players/YouTube'

import { VIDEO_SOURCES } from '../data';

const Wrapper = styled.div`
  max-width: 100vw;
  height: 100%;
`

const VideoPlayer = forwardRef(({ lesson, startAt }, ref) => {
  let url = `https://www.youtube.com/embed/${VIDEO_SOURCES[lesson]}`;

  if (startAt) {
    url = `${url}?t=${startAt}`
  }
  
  return (
    <Wrapper>
      <YouTubePlayer
        ref={ref}
        url={url}
        controls
        width="100%"
        height="100%"
        config={{ youtube: { playerVars: { playsinline: 1, modestbranding: 1 } } }} />
    </Wrapper>
  )
})

VideoPlayer.propTypes = {
  lesson: PropTypes.number.isRequired,
  startAt: PropTypes.number
};

export default VideoPlayer;
