// src/App.tsx
import { useState } from "react";

export default function MyWallet() {
    const [connected, setConnected] = useState(false);
    const [publicKey, setPublicKey] = useState<string | null>(null);
    const [balance, setBalance] = useState<string | null>(null);

    const handleConnect = () => {
        // Simüle edilmiş bağlantı işlemi
        const mockPubKey = "GAHACKATHONDEMOFAKEPUBKEYEXAMPLE...";
        setPublicKey(mockPubKey);
        setConnected(true);
        setBalance("12.537 XLM");
    };

    return (
        <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center space-y-6 p-6">
            <h1 className="text-3xl font-bold">🧠 NeuroWallet</h1>

            {!connected ? (
                <button
                    onClick={handleConnect}
                    className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-2xl text-lg shadow-md"
                >
                    Cüzdanı Bağla
                </button>
            ) : (
                <div className="space-y-4 text-center">
                    <p className="text-sm text-gray-400">Bağlı cüzdan:</p>
                    <p className="text-md font-mono bg-gray-800 p-2 rounded-xl">
                        {publicKey}
                    </p>
                    <p className="text-lg font-semibold">Bakiye: {balance}</p>

                    <div className="grid grid-cols-2 gap-4 mt-4">
                        <button className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-xl">
                            Ödeme Gönder
                        </button>
                        <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-xl">
                            DEX İşlemi Yap
                        </button>
                        <button className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-xl col-span-2">
                            Sinyal ile İşlem
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
