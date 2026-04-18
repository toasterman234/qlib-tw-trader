import { useState, useEffect, useCallback } from 'react'
import {
  datasetsApi,
  universeApi,
  syncApi,
  DatasetInfo,
  TestResult,
  CategoryInfo,
  StockInfo,
  SyncStatusResponse,
  MonthlyStatusResponse,
  SyncAllResponse,
} from '@/api/client'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Loader2,
  CheckCircle,
  XCircle,
  Play,
  RefreshCw,
  Wrench,
  Download,
  Database,
} from 'lucide-react'
import { useFetchOnChange } from '@/hooks/useFetchOnChange'

const statusClasses: Record<string, string> = {
  available: 'bg-green-500 text-white',
  needs_accumulation: 'bg-yellow-500 text-white',
  not_implemented: 'bg-gray-400 text-white',
  pending: 'bg-slate-400 text-white',
}

const statusLabels: Record<string, string> = {
  available: 'Available',
  needs_accumulation: 'Needs Accumulation',
  not_implemented: 'Not Implemented',
  pending: 'Planned',
}

const isDateStale = (dateStr: string | null): boolean => {
  if (!dateStr) return true
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  const checkDate = new Date(dateStr)
  return checkDate < yesterday
}

const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  return 'Unknown error'
}

const checkSyncResponse = (result: SyncAllResponse, datasetName: string): void => {
  if (!result.errors || result.errors.length === 0) return
  const errorMsgs = result.errors.slice(0, 3).map(e => `${e.stock_id}: ${e.error}`).join('\n')
  const moreCount = result.errors.length > 3 ? `\n...and ${result.errors.length - 3} more errors` : ''
  alert(`${datasetName} finished with ${result.errors.length} errors:\n${errorMsgs}${moreCount}`)
}

type SyncPanelConfig = {
  type: 'daily' | 'monthly'
  title: string
  status: SyncStatusResponse | MonthlyStatusResponse | null
  syncAction?: () => Promise<void>
  repairAction?: () => Promise<void>
  syncLabel?: string
  repairLabel?: string
}

function getDailySummary(status: SyncStatusResponse | null) {
  const totalStocks = status?.stocks.length || 0
  const completeStocks = status?.stocks.filter(s => s.coverage_pct >= 95 && s.latest_date && !isDateStale(s.latest_date)).length || 0
  const earliestDate = status?.stocks.filter(s => s.earliest_date).map(s => s.earliest_date!).sort()[0] || 'No data'
  const latestDate = status?.stocks.filter(s => s.latest_date).map(s => s.latest_date!).sort().reverse()[0] || 'No data'
  return {
    totalStocks,
    completeStocks,
    days: status?.trading_days || 0,
    earliestDate,
    latestDate,
  }
}

function getMonthlySummary(status: MonthlyStatusResponse | null) {
  const totalStocks = status?.stocks.length || 0
  const completeStocks = status?.stocks.filter(s => s.coverage_pct >= 95).length || 0
  const earliestDate = status?.stocks.filter(s => s.earliest_month).map(s => s.earliest_month!).sort()[0] || 'No data'
  const latestDate = status?.stocks.filter(s => s.latest_month).map(s => s.latest_month!).sort().reverse()[0] || 'No data'
  return {
    totalStocks,
    completeStocks,
    days: status?.expected_months || 0,
    earliestDate,
    latestDate,
  }
}

function coverageBarClass(coverage: number, latestDate?: string | null) {
  if (coverage === 0) return 'bg-gray-300'
  if (!latestDate) return coverage >= 95 ? 'bg-green-500' : 'bg-yellow-500'
  return coverage >= 95 && !isDateStale(latestDate) ? 'bg-green-500' : 'bg-yellow-500'
}

export function Datasets() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [categories, setCategories] = useState<CategoryInfo[]>([])
  const [universe, setUniverse] = useState<StockInfo[]>([])
  const [universeName, setUniverseName] = useState('Universe')
  const [universeDescription, setUniverseDescription] = useState('')
  const [universeUpdatedAt, setUniverseUpdatedAt] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [syncingUniverse, setSyncingUniverse] = useState(false)
  const [testResults, setTestResults] = useState<Record<string, TestResult>>({})
  const [testingDataset, setTestingDataset] = useState<string | null>(null)
  const [stockId, setStockId] = useState('')

  const [stockDailyStatus, setStockDailyStatus] = useState<SyncStatusResponse | null>(null)
  const [perStatus, setPerStatus] = useState<SyncStatusResponse | null>(null)
  const [institutionalStatus, setInstitutionalStatus] = useState<SyncStatusResponse | null>(null)
  const [marginStatus, setMarginStatus] = useState<SyncStatusResponse | null>(null)
  const [adjStatus, setAdjStatus] = useState<SyncStatusResponse | null>(null)
  const [shareholdingStatus, setShareholdingStatus] = useState<SyncStatusResponse | null>(null)
  const [securitiesLendingStatus, setSecuritiesLendingStatus] = useState<SyncStatusResponse | null>(null)
  const [monthlyRevenueStatus, setMonthlyRevenueStatus] = useState<MonthlyStatusResponse | null>(null)

  const loadData = useCallback(async () => {
    try {
      const [datasetsRes, categoriesRes, universeRes, allSyncRes] = await Promise.all([
        datasetsApi.list(),
        datasetsApi.categories(),
        universeApi.get(),
        syncApi.allStatus(),
      ])
      setDatasets(datasetsRes.datasets)
      setCategories(categoriesRes.categories)
      setUniverse(universeRes.stocks)
      setUniverseName(universeRes.name)
      setUniverseDescription(universeRes.description)
      setUniverseUpdatedAt(universeRes.updated_at)
      setStockDailyStatus(allSyncRes.stock_daily)
      setPerStatus(allSyncRes.per)
      setInstitutionalStatus(allSyncRes.institutional)
      setMarginStatus(allSyncRes.margin)
      setAdjStatus(allSyncRes.adj)
      setShareholdingStatus(allSyncRes.shareholding)
      setSecuritiesLendingStatus(allSyncRes.securities_lending)
      setMonthlyRevenueStatus(allSyncRes.monthly_revenue)
      if (!stockId && universeRes.stocks.length > 0) {
        setStockId(universeRes.stocks[0].stock_id)
      }
    } catch (error) {
      console.error('Failed to load datasets page:', error)
    } finally {
      setLoading(false)
    }
  }, [stockId])

  useEffect(() => {
    loadData()
  }, [loadData])

  useFetchOnChange('datasets', loadData)

  const refreshSyncStatus = async () => {
    try {
      const allSyncRes = await syncApi.allStatus()
      setStockDailyStatus(allSyncRes.stock_daily)
      setPerStatus(allSyncRes.per)
      setInstitutionalStatus(allSyncRes.institutional)
      setMarginStatus(allSyncRes.margin)
      setAdjStatus(allSyncRes.adj)
      setShareholdingStatus(allSyncRes.shareholding)
      setSecuritiesLendingStatus(allSyncRes.securities_lending)
      setMonthlyRevenueStatus(allSyncRes.monthly_revenue)
    } catch (error) {
      console.error('Failed to refresh sync status:', error)
    }
  }

  const syncUniverse = async () => {
    setSyncingUniverse(true)
    try {
      await universeApi.sync()
      const universeRes = await universeApi.get()
      setUniverse(universeRes.stocks)
      setUniverseName(universeRes.name)
      setUniverseDescription(universeRes.description)
      setUniverseUpdatedAt(universeRes.updated_at)
      if (!stockId && universeRes.stocks.length > 0) {
        setStockId(universeRes.stocks[0].stock_id)
      }
    } catch (error) {
      alert(`Universe sync failed: ${getErrorMessage(error)}`)
    } finally {
      setSyncingUniverse(false)
    }
  }

  const testDataset = async (datasetName: string, requiresStockId: boolean) => {
    setTestingDataset(datasetName)
    try {
      const result = await datasetsApi.test(datasetName, requiresStockId ? stockId : undefined, 7)
      setTestResults(prev => ({ ...prev, [datasetName]: result }))
    } catch (error) {
      setTestResults(prev => ({
        ...prev,
        [datasetName]: {
          dataset: datasetName,
          success: false,
          record_count: 0,
          sample_data: null,
          error: getErrorMessage(error),
        },
      }))
    } finally {
      setTestingDataset(null)
    }
  }

  const handleStockDailySync = async () => {
    await syncApi.bulk()
    await refreshSyncStatus()
  }
  const handleStockDailyRepair = async () => {
    await syncApi.calendar('2020-01-01')
    const result = await syncApi.all('2020-01-01')
    checkSyncResponse(result, 'Market price data')
    await refreshSyncStatus()
  }
  const handleAdjSync = async () => {
    await syncApi.adjBulk()
    await refreshSyncStatus()
  }
  const handleAdjRepair = async () => {
    await syncApi.calendar('2020-01-01')
    const result = await syncApi.adjAll('2020-01-01')
    checkSyncResponse(result, 'Adjusted close data')
    await refreshSyncStatus()
  }
  const handlePerSync = async () => {
    await syncApi.perBulk()
    await refreshSyncStatus()
  }
  const handlePerRepair = async () => {
    await syncApi.calendar('2020-01-01')
    const result = await syncApi.perAll('2020-01-01')
    checkSyncResponse(result, 'Valuation data')
    await refreshSyncStatus()
  }
  const handleInstitutionalSync = async () => {
    await syncApi.institutionalBulk()
    await refreshSyncStatus()
  }
  const handleInstitutionalRepair = async () => {
    await syncApi.calendar('2020-01-01')
    const result = await syncApi.institutionalAll('2020-01-01')
    checkSyncResponse(result, 'Institutional flow')
    await refreshSyncStatus()
  }
  const handleMarginSync = async () => {
    await syncApi.marginBulk()
    await refreshSyncStatus()
  }
  const handleMarginRepair = async () => {
    await syncApi.calendar('2020-01-01')
    const result = await syncApi.marginAll('2020-01-01')
    checkSyncResponse(result, 'Margin / short data')
    await refreshSyncStatus()
  }
  const handleShareholdingSync = async () => {
    await syncApi.shareholdingBulk()
    await refreshSyncStatus()
  }
  const handleShareholdingRepair = async () => {
    await syncApi.calendar('2020-01-01')
    const result = await syncApi.shareholdingAll('2020-01-01')
    checkSyncResponse(result, 'Ownership data')
    await refreshSyncStatus()
  }
  const handleSecuritiesLendingRepair = async () => {
    await syncApi.calendar('2020-01-01')
    const result = await syncApi.securitiesLendingAll('2020-01-01')
    checkSyncResponse(result, 'Securities lending data')
    await refreshSyncStatus()
  }
  const handleMonthlyRevenueRepair = async () => {
    const result = await syncApi.monthlyRevenueAll(2020)
    checkSyncResponse(result, 'Revenue data')
    await refreshSyncStatus()
  }

  const syncConfigByDataset: Record<string, SyncPanelConfig> = {
    USEquityPrice: { type: 'daily', title: 'Market Price Data', status: stockDailyStatus, syncAction: handleStockDailySync, repairAction: handleStockDailyRepair, syncLabel: 'Sync', repairLabel: 'Repair' },
    USEquityPriceAdj: { type: 'daily', title: 'Adjusted Close Data', status: adjStatus, syncAction: handleAdjSync, repairAction: handleAdjRepair, syncLabel: 'Sync', repairLabel: 'Repair' },
    USEquityValuation: { type: 'daily', title: 'Valuation Data', status: perStatus, syncAction: handlePerSync, repairAction: handlePerRepair, syncLabel: 'Sync', repairLabel: 'Repair' },
    USInstitutionalFlow: { type: 'daily', title: 'Institutional Flow', status: institutionalStatus, syncAction: handleInstitutionalSync, repairAction: handleInstitutionalRepair, syncLabel: 'Sync', repairLabel: 'Repair' },
    USShortInterest: { type: 'daily', title: 'Margin / Short Data', status: marginStatus, syncAction: handleMarginSync, repairAction: handleMarginRepair, syncLabel: 'Sync', repairLabel: 'Repair' },
    USOwnership: { type: 'daily', title: 'Ownership Data', status: shareholdingStatus, syncAction: handleShareholdingSync, repairAction: handleShareholdingRepair, syncLabel: 'Sync', repairLabel: 'Repair' },
    USRevenue: { type: 'monthly', title: 'Revenue Data', status: monthlyRevenueStatus, repairAction: handleMonthlyRevenueRepair, repairLabel: 'Repair' },
  }

  const groupedDatasets = datasets.reduce((acc, ds) => {
    if (!acc[ds.category]) acc[ds.category] = []
    acc[ds.category].push(ds)
    return acc
  }, {} as Record<string, DatasetInfo[]>)

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Datasets</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Inspect the current US market data catalog, test adapters, and monitor sync coverage.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-muted-foreground">Test symbol</label>
          <input
            type="text"
            value={stockId}
            onChange={(e) => setStockId(e.target.value.toUpperCase())}
            className="w-28 px-2 py-1 text-sm border rounded bg-background"
            placeholder="AAPL"
          />
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              <span>{universeName}</span>
              <span className="ml-2 px-2 py-0.5 text-xs rounded bg-blue-500 text-white">{universe.length} symbols</span>
            </div>
            <button
              className="flex items-center gap-1 px-3 py-1 text-sm border rounded hover:bg-secondary disabled:opacity-50"
              onClick={syncUniverse}
              disabled={syncingUniverse}
            >
              <RefreshCw className={`h-4 w-4 ${syncingUniverse ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground mb-3">{universeDescription}</div>
          {universeUpdatedAt && (
            <div className="text-xs text-muted-foreground mb-3">
              Updated: {new Date(universeUpdatedAt).toLocaleString()}
            </div>
          )}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 max-h-56 overflow-y-auto">
            {universe.map((stock) => (
              <button
                key={stock.stock_id}
                className="text-left border rounded px-3 py-2 hover:border-primary hover:bg-secondary/40"
                onClick={() => setStockId(stock.stock_id)}
              >
                <div className="font-mono font-semibold">{stock.stock_id}</div>
                <div className="text-xs text-muted-foreground truncate">{stock.name}</div>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {categories.map((cat) => (
          <Card key={cat.id}>
            <CardContent className="pt-4">
              <div className="text-sm text-muted-foreground">{cat.name}</div>
              <div className="text-2xl font-bold">{cat.available}/{cat.total}</div>
              <div className="text-xs text-muted-foreground">available / total</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="space-y-5">
        {Object.entries(groupedDatasets).map(([category, items]) => (
          <Card key={category}>
            <CardHeader>
              <CardTitle className="text-lg">{categories.find(c => c.id === category)?.name || category}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {items.map((dataset) => (
                  <DatasetRow
                    key={dataset.name}
                    dataset={dataset}
                    testResult={testResults[dataset.name]}
                    isTesting={testingDataset === dataset.name}
                    onTest={() => testDataset(dataset.name, dataset.requires_stock_id)}
                    panelConfig={syncConfigByDataset[dataset.name] || null}
                  />
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

type DatasetRowProps = {
  dataset: DatasetInfo
  testResult?: TestResult
  isTesting: boolean
  onTest: () => void
  panelConfig: SyncPanelConfig | null
}

function DatasetRow({ dataset, testResult, isTesting, onTest, panelConfig }: DatasetRowProps) {
  const [expanded, setExpanded] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [repairing, setRepairing] = useState(false)

  const runSync = async () => {
    if (!panelConfig?.syncAction) return
    setSyncing(true)
    try {
      await panelConfig.syncAction()
    } catch (error) {
      alert(`Sync failed: ${getErrorMessage(error)}`)
    } finally {
      setSyncing(false)
    }
  }

  const runRepair = async () => {
    if (!panelConfig?.repairAction) return
    setRepairing(true)
    try {
      await panelConfig.repairAction()
    } catch (error) {
      alert(`Repair failed: ${getErrorMessage(error)}`)
    } finally {
      setRepairing(false)
    }
  }

  const dailySummary = panelConfig?.type === 'daily' ? getDailySummary(panelConfig.status as SyncStatusResponse | null) : null
  const monthlySummary = panelConfig?.type === 'monthly' ? getMonthlySummary(panelConfig.status as MonthlyStatusResponse | null) : null
  const barItems = panelConfig?.type === 'daily'
    ? (panelConfig.status as SyncStatusResponse | null)?.stocks || []
    : (panelConfig?.type === 'monthly' ? (panelConfig.status as MonthlyStatusResponse | null)?.stocks || [] : [])

  return (
    <div className="border rounded-lg p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <span className={`px-2 py-0.5 text-xs rounded ${statusClasses[dataset.status] || 'bg-slate-400 text-white'}`}>
            {statusLabels[dataset.status] || dataset.status}
          </span>
          <div>
            <div className="font-medium">{dataset.display_name}</div>
            <div className="text-sm text-muted-foreground font-mono">{dataset.name}</div>
            <div className="text-xs text-muted-foreground mt-1">{dataset.source}</div>
            {dataset.description && (
              <div className="text-xs text-muted-foreground mt-2">{dataset.description}</div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap justify-end">
          {dataset.status === 'available' && (
            <button
              className="flex items-center gap-1 px-3 py-1 text-sm border rounded hover:bg-secondary disabled:opacity-50"
              onClick={onTest}
              disabled={isTesting}
            >
              {isTesting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              Test
            </button>
          )}
          {panelConfig?.syncAction && (
            <button
              className="flex items-center gap-1 px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
              onClick={runSync}
              disabled={syncing || repairing}
            >
              {syncing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
              {panelConfig.syncLabel || 'Sync'}
            </button>
          )}
          {panelConfig?.repairAction && (
            <button
              className="flex items-center gap-1 px-3 py-1 text-sm bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50"
              onClick={runRepair}
              disabled={syncing || repairing}
            >
              {repairing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wrench className="h-4 w-4" />}
              {panelConfig.repairLabel || 'Repair'}
            </button>
          )}
        </div>
      </div>

      {testResult && (
        <div className={`mt-3 p-3 rounded border ${testResult.success ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <div className="flex items-center gap-2">
            {testResult.success ? <CheckCircle className="h-4 w-4 text-green-600" /> : <XCircle className="h-4 w-4 text-red-600" />}
            <span className={`text-sm font-medium ${testResult.success ? 'text-green-700' : 'text-red-700'}`}>
              {testResult.success ? `Test succeeded · ${testResult.record_count} records` : testResult.error || 'Test failed'}
            </span>
            {testResult.sample_data && (
              <button className="ml-auto text-xs underline" onClick={() => setExpanded(!expanded)}>
                {expanded ? 'Hide sample' : 'Show sample'}
              </button>
            )}
          </div>
          {expanded && testResult.sample_data && (
            <pre className="mt-3 text-xs bg-white p-3 rounded overflow-auto">{JSON.stringify(testResult.sample_data, null, 2)}</pre>
          )}
        </div>
      )}

      {panelConfig && (
        <div className="mt-4 space-y-3">
          <div className="text-sm font-medium">{panelConfig.title}</div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="p-3 bg-secondary/40 rounded border">
              <div className="text-xs text-muted-foreground">Periods</div>
              <div className="text-lg font-bold">{(dailySummary || monthlySummary)?.days || 0}</div>
            </div>
            <div className="p-3 bg-secondary/40 rounded border">
              <div className="text-xs text-muted-foreground">Complete Symbols</div>
              <div className="text-lg font-bold">
                {(dailySummary || monthlySummary)?.completeStocks || 0}/{(dailySummary || monthlySummary)?.totalStocks || 0}
              </div>
            </div>
            <div className="p-3 bg-secondary/40 rounded border">
              <div className="text-xs text-muted-foreground">Earliest</div>
              <div className="text-sm font-mono font-semibold">{(dailySummary || monthlySummary)?.earliestDate || 'No data'}</div>
            </div>
            <div className="p-3 bg-secondary/40 rounded border">
              <div className="text-xs text-muted-foreground">Latest</div>
              <div className="text-sm font-mono font-semibold">{(dailySummary || monthlySummary)?.latestDate || 'No data'}</div>
            </div>
          </div>

          {barItems.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs text-muted-foreground">Coverage preview</span>
              </div>
              <div className="flex gap-0.5 flex-wrap">
                {barItems.slice(0, 100).map((item) => (
                  <div
                    key={item.stock_id}
                    className={`w-2 h-4 rounded-sm ${coverageBarClass(item.coverage_pct, 'latest_date' in item ? item.latest_date : null)}`}
                    title={`${item.stock_id} ${item.name}: ${item.coverage_pct}%`}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
