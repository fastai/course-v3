import React from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import YouTubePlayer from 'react-player/lib/players/YouTube'

const VIDEO_SOURCES = {
  1: "XfoYk_Z5AkI",
  2: "ccMHJeQU4Qw",
  3: "MpZxV6DVsmM",
  4: "9YK6AnqpuEA",
  5: "CJKnDu2dxOE",
  6: "hkBa9pU-H48",
  7: "DGdRC4h78_o",
};

const Wrapper = styled.div`
  max-width: 100vw;
  height: 99vh;
`

const VideoPlayer = React.forwardRef((props, ref) => {
  // "https://www.youtube.com/embed/XfoYk_Z5AkI",
  const { lesson } = props;
  return (
    <Wrapper>
      <YouTubePlayer ref={ref} url={`https://www.youtube.com/embed/${VIDEO_SOURCES[lesson]}`} controls width="100%" height="100%" />
    </Wrapper>
  )
})

VideoPlayer.propTypes = {
  lesson: PropTypes.number.isRequired,
};

export default VideoPlayer;
