import { BrowserRouter, Routes, Route } from 'react-router-dom'
import HomePage from './HomePage'
import UFCPage from './pages/ufc/UFCPage'  // Changed from './pages/ufc/UFCPage'
// import { Captcha } from './Captcha'

function App() {
  return (
    // <Captcha>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/ufc" element={<UFCPage />} />
        </Routes>
      </BrowserRouter>
    // </Captcha>
  )
}

export default App