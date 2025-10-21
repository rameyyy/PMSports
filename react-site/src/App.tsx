import { BrowserRouter, Routes, Route } from 'react-router-dom'
import HomePage from './HomePage'
import UFCPage from './pages/ufc/UFCPage'  // Changed from './pages/ufc/UFCPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/ufc" element={<UFCPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App