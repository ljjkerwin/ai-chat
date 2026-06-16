import { RAGProvider } from '@/contexts/RAGContext'
import { SessionProvider } from '@/contexts/SessionContext'
import Sidebar from '@/components/layout/Sidebar'

export default function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <RAGProvider>
      <SessionProvider>
        <div className="flex h-full">
          <Sidebar />
          <main className="flex-1 flex flex-col min-w-0 h-full">
            {children}
          </main>
        </div>
      </SessionProvider>
    </RAGProvider>
  )
}
