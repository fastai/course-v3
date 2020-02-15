import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';
import { AiOutlineMenuFold, AiOutlineMenuUnfold } from 'react-icons/ai';

import Panel from './Panel';
import ToggleBtn from './ToggleBtn';

const Container = styled.div`
  display: flex;
  height: 100vh;
  height: -webkit-fill-available;
  flex-direction: row;
  width: 100vw;
`;

const MainContentContainer = styled.main`
  width: 100%;
  background: #222;
  position: relative;
`;

const MainContent = styled.div`
  position: absolute;
  width: 100%;
  overscroll-behavior: none;
  height: 100%;
  @media (max-width: 950px) {
    position: fixed;
  }
`;


const LeftToggleBtn = (props) => (
  <ToggleBtn
    style={{ top: '52px', left: '12px'}}
    ShownIcon={AiOutlineMenuFold}
    HiddenIcon={AiOutlineMenuUnfold}
    {...props} />
);

const RightToggleBtn = (props) => (
  <ToggleBtn
    style={{ top: '52px', right: '12px'}}
    ShownIcon={AiOutlineMenuUnfold}
    HiddenIcon={AiOutlineMenuFold}
    {...props} />
);

const Layout = ({ LeftPanelContent, RightPanelContent, children }) => {
  const [leftOpened, setLeftOpened] = useState(true);
  const [rightOpened, setRightOpened] = useState(true);
  
  const toggleLeft = useCallback(() => setLeftOpened(!leftOpened), [leftOpened]);
  const toggleRight = useCallback(() => setRightOpened(!rightOpened), [rightOpened]);

  return (
    <Container>
      <Panel shown={leftOpened} width={'12rem'}>
        { LeftPanelContent }
      </Panel>
      <MainContentContainer>
        <LeftToggleBtn shown={leftOpened} onClick={toggleLeft} />
        <MainContent>
          { children }
        </MainContent>
        <RightToggleBtn shown={rightOpened} onClick={toggleRight} />
      </MainContentContainer>
      <Panel shown={rightOpened} width={'35rem'}>
        { RightPanelContent }
      </Panel>
    </Container>
  )
}

Layout.propTypes = {
  LeftPanelContent: PropTypes.element,
  RightPanelContent: PropTypes.element,
  children: PropTypes.element
}

export default Layout
