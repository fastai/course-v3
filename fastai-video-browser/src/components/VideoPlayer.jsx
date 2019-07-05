import React from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import YouTubePlayer from 'react-player/lib/players/YouTube'

const VIDEO_SOURCES = {
  1: "XfoYk_Z5AkI",
  2: "ccMHJeQU4Qw",
  3: "MpZxV6DVsmM",
  4: "qqt3aMPB81c",
  5: "CJKnDu2dxOE",
  6: "hkBa9pU-H48",
  7: "9spwoDYwW_I",
  8:  "4u8FxNEDUeg",
  9:  "AcA8HAYh7IE",
  10: "HR0lt1hlR6U",
  11: "hPQKzsjTyyQ",
  12: "vnOpEwmtFJ8",
  13: "3TqN_M1L4ts",
  14: "8wd8zFzTG38",
};

const Wrapper = styled.div`
  max-width: 100vw;
  height: 99vh;
`

const VideoPlayer = React.forwardRef((props, ref) => {
  const { lesson, startAt } = props;
  let url = `https://www.youtube.com/embed/${VIDEO_SOURCES[lesson]}`;

  if (startAt) {
    url = `${url}?t=${startAt}`
  }
  
  return (
    <Wrapper>
      <YouTubePlayer ref={ref} url={url} controls width="100%" height="100%" />
    </Wrapper>
  )
})

VideoPlayer.propTypes = {
  lesson: PropTypes.number.isRequired,
  startAt: PropTypes.number
};

export default VideoPlayer;
