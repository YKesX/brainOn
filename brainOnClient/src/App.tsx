import React, { useEffect, useState } from "react";
import { useEEG } from "./context/EEGContext";
import { fetchBalance, sendPayment } from "./utils/walletUtils";
import toast, { Toaster } from 'react-hot-toast';
import { FiExternalLink } from "react-icons/fi";
import { Asset } from "@stellar/stellar-sdk";
import { dexBuy } from "./utils/dexPayment";

//curl -X POST http://localhost:5000/emit-signal -H "Content-Type: application/json" -d '{"signal": "0100"}'

function App() {

  const { isSendLoading, balance, latestTx, isDexSendLoading } = useEEG();

  const [connected, setConnected] = useState(false);
  const [publicKey, setPublicKey] = useState<string | null>(null);
  const [balanceThis, setBalance] = useState<string | null>(null);
  const [isSendLoadingThis, setIsSendLoading] = useState(false);
  const [isDexSendLoadingThis, setIsDexSendLoading] = useState(false);

  const [latestTxId, setLatestTxId] = useState("")

  const handleConnect = async () => {
    const mockPubKey = "GB3JDFW5TXE4TBWWN75J7AJHG5FBP74...";
    setPublicKey(mockPubKey);
    setConnected(true);
    const balance = await fetchBalance('GB3U4B7MURE25RDOODZLSRSMIOIQNMGI2GDHX7HGBSVW4T74XWXTGFOK')
    setBalance(balance);
  };

  const handleSend = async () => {
    try {
      setIsSendLoading(true)
      const resHash = await sendPayment(
        "PRIVATE_KEY",
        "GB3JDFW5TXE4TBWWN75J7AJHG5FBP74J35CO33KQ5QTMMX3KIBNO7JRA",
        "2.55"
      );
      setIsSendLoading(false)
      toast.success("✅ Payment successfully sent!");
      const newBalance = await fetchBalance('GB3U4B7MURE25RDOODZLSRSMIOIQNMGI2GDHX7HGBSVW4T74XWXTGFOK')
      setBalance(newBalance)
      setLatestTxId(resHash.hash)
    } catch (e) {
      setIsSendLoading(false)
      toast.error("❌ İşlem sırasında bir hata oluştu.");

      console.log(e.error)
    }
  }

  const sendDexTx = async () => {
    try {
      setIsDexSendLoading(true)
      const SOURCE_SECRET =
        "PRIVATE_KEY";

      const sellingAsset = Asset.native();                // XLM
      const buyingAsset = new Asset(                     // USDC
        "USDC",
        "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
      );

      const buyAmount = "1.5";           // 50 USDC al
      // const price = { n: 15, d: 100 }; // 0.15 XLM / USDC

      const res = await dexBuy(SOURCE_SECRET, sellingAsset, buyingAsset, buyAmount, "4.12");
      toast.success("✅ Tx successfully sent!");
      setLatestTxId(res.hash)
      console.log(res)
      setIsDexSendLoading(false)
    } catch (error) {
      toast.error("❌ Something went wrong!");
      console.error(error);
    }
    finally {
      setIsDexSendLoading(false);
    }

  }

  useEffect(() => { setIsSendLoading(isSendLoading) }, [isSendLoading])
  useEffect(() => { setBalance(balance) }, [balance])
  useEffect(() => { setLatestTxId(latestTx) }, [latestTx])
  useEffect(() => { setIsDexSendLoading(isDexSendLoading) }, [isDexSendLoading])

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center px-4 py-8">
      <Toaster
        position="top-center"
        toastOptions={{
          style: {
            background: '#1f2937',
            color: '#f9fafb',
            border: '1px solid #374151',
            borderRadius: '12px',
            fontSize: '14px',
            fontWeight: '500',
          },
        }}
      />

      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-white rounded-2xl shadow-sm border border-gray-200 mb-6">
            <span className="text-2xl">🧠</span>
          </div>
          <h1 className="text-3xl font-semibold text-gray-900 tracking-tight mb-2">BrainOn</h1>
          <p className="text-gray-500 text-sm">Neural-powered web3 wallet</p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-3xl shadow-sm border border-gray-200 p-8">
          {!connected ? (
            <div className="text-center space-y-6">
              <div className="space-y-2">
                <h2 className="text-xl font-medium text-gray-900">Connect Wallet</h2>
                <p className="text-gray-500 text-sm">Connect your wallet to get started</p>
              </div>

              <button
                onClick={handleConnect}
                className="w-full bg-gray-900 text-white rounded-2xl py-4 px-6 font-medium transition-all duration-200 hover:bg-gray-800 hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-gray-900 focus:ring-offset-2 active:scale-[0.98]"
              >
                Connect Wallet
              </button>
            </div>
          ) : (
            <div className="space-y-8">
              {/* Wallet Info */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-medium text-gray-900">Wallet</h2>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-green-600 font-medium">Connected</span>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-2xl p-4 space-y-3">
                  <div>
                    <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Public Key</p>
                    <p className="font-mono text-sm text-gray-700 break-all">{publicKey}</p>
                  </div>

                  <div>
                    <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Balance</p>
                    <p className="text-2xl font-semibold text-gray-900">{balanceThis}</p>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="space-y-3">
                <p className="text-sm font-medium text-gray-900 mb-4">Actions</p>

                <button
                  onClick={handleSend}
                  disabled={isSendLoadingThis}
                  className="w-full bg-gray-900 text-white rounded-2xl py-4 px-6 font-medium transition-all duration-200 hover:bg-gray-800 hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-gray-900 focus:ring-offset-2 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:bg-gray-900"
                >
                  {isSendLoadingThis ? (
                    <div className="flex items-center justify-center space-x-2">
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      <span>Sending...</span>
                    </div>
                  ) : (
                    "Transfer XLM"
                  )}
                </button>

                <button
                  onClick={sendDexTx}
                  disabled={isDexSendLoadingThis}
                  className="w-full bg-white text-gray-900 rounded-2xl py-4 px-6 font-medium border border-gray-200 transition-all duration-200 hover:bg-gray-50 hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-gray-900 focus:ring-offset-2 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:bg-white"
                >
                  {isDexSendLoadingThis ? (
                    <div className="flex items-center justify-center space-x-2">
                      <div className="w-4 h-4 border-2 border-gray-300 border-t-gray-900 rounded-full animate-spin"></div>
                      <span>Processing...</span>
                    </div>
                  ) : (
                    "DEX Transaction"
                  )}
                </button>
              </div>

              {/* Transaction History */}
              {latestTxId && (
                <div className="pt-6 border-t border-gray-100">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-sm font-medium text-gray-900">Recent Transaction</p>
                  </div>

                  <div className="bg-gray-50 rounded-2xl p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Transaction Hash</p>
                        <p className="font-mono text-sm text-gray-700 break-all pr-2">{latestTxId}</p>
                      </div>
                      <a
                        href={`https://stellar.expert/explorer/public/tx/${latestTxId}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex-shrink-0 inline-flex items-center justify-center w-8 h-8 bg-white rounded-lg border border-gray-200 text-gray-500 hover:text-gray-700 hover:border-gray-300 transition-colors duration-200"
                      >
                        <FiExternalLink className="w-4 h-4" />
                      </a>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-xs text-gray-400">Powered by neural signals & Stellar blockchain</p>
        </div>
      </div>
    </div>
  );
}

export default App;