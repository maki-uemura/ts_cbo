import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import InquiryScreen from './components/InquiryScreen';

const App = () => {
  return (
    <Router>
      <div>
        <Switch>
          <Route path="/inquiry" component={InquiryScreen} />
          {/* Add other routes here */}
        </Switch>
      </div>
    </Router>
  );
};

export default App;