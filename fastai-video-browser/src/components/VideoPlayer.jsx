import React from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import YouTubePlayer from 'react-player/lib/players/YouTube'

const VIDEO_SOURCES = [
  "XfoYk_Z5AkI",
  "ccMHJeQU4Qw",
  "MpZxV6DVsmM",
  "9YK6AnqpuEA",
  "CJKnDu2dxOE",
  "hkBa9pU-H48",
  "DGdRC4h78_o",
];

const Wrapper = styled.div`
  flex: 5;
  height: 80vh;
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
