import React, { useCallback } from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';
import { AiOutlineMenuFold, AiOutlineMenuUnfold } from 'react-icons/ai';
import { useLocalStorage } from 'react-use';

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
  const [panelState, setPanelState] = useLocalStorage('panel-state', { left: true, right: true })
  
  const toggleLeft = useCallback(() => setPanelState({ ...panelState, left: !panelState.left }), [panelState, setPanelState]);
  const toggleRight = useCallback(() => setPanelState({ ...panelState, right: !panelState.right }), [panelState, setPanelState]);

  return (
    <Container>
      <Panel shown={panelState.left} width={'12rem'}>
        { LeftPanelContent }
      </Panel>
      <MainContentContainer>
        <LeftToggleBtn shown={panelState.left} onClick={toggleLeft} />
        <MainContent>
          { children }
        </MainContent>
        <RightToggleBtn shown={panelState.right} onClick={toggleRight} />
      </MainContentContainer>
      <Panel shown={panelState.right} width={'35rem'}>
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
