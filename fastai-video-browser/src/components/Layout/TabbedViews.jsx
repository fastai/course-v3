import React, { useState } from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';

import { standard } from '../../utils/easing';

const Container = styled.div`
  position: relative;
  display: flex;
  flex-direction: column;
  max-height: 100vh;
`;

const ContentContainer = styled.div`
  max-height: 100%;
  flex: 1;
  /*overflow: scroll;*/
`;

const TabContainer = styled.div`
  width: 100%;
  display: flex;
  flex: 1;
  flex-direction: row;
  align-items: stretch;
  flex-shrink: 0;
  flex-grow: 0;
  height: 55px;
  background: #fff;
`;

const Tab = styled.div`
  height: 65px;
  display: inline-flex;
  cursor: pointer;
  flex: 1;
  height: 100%;
  align-items: center;
  justify-content: center;
  position: relative;
  color: var(--text-dark);
  opacity: 0.8;
  &, &::after {
    transition: all 0.3s ${standard};
  }
  &::after {
    content: "";
    width: 100%;
    height: 2px;
    background: rgba(0, 0, 0, 0.15);
    position: absolute;
    bottom: 0;
    left: 0;
  }
  ${({ active }) => active ? `
    font-weight: bold;
    opacity: 1.0;
    &::after {
      background: var(--fastai-blue);
    }
  ` : `
    &:hover {
      &::after {
        background: rgba(0, 0, 0, 0.3);
      }
    }
  `}
`;

const TabbedViews = ({ children }) => {
  const [openedTab, setOpenedTab] = useState(0);

  return (
    <Container>
      <TabContainer>
        <Tab onClick={() => setOpenedTab(0)} active={openedTab === 0}>
          Notes
        </Tab>
        <Tab onClick={() => setOpenedTab(1)} active={openedTab === 1}>
          Transcript Search
        </Tab>
      </TabContainer>
      <ContentContainer>
        {React.Children.map(children, (child, idx) => (
          idx === openedTab ? child: null 
        ))}
      </ContentContainer>
    </Container>
  )
}

TabbedViews.propTypes = {
  children: PropTypes.arrayOf(PropTypes.element)
}

export default TabbedViews;
