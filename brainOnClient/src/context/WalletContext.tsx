import { createContext, useContext, useState, useEffect } from "react";
import freighterApi, {
    isConnected,
    isAllowed,
    requestAccess,
    getAddress,
} from "@stellar/freighter-api";
interface WalletContextType {
    publicKey: string | null;
    connectWallet: () => Promise<void>;
}

const WalletContext = createContext<WalletContextType | undefined>(undefined);

export const WalletProvider = ({ children }: { children: React.ReactNode }) => {
    const [publicKey, setPublicKey] = useState<string | null>(null);

    useEffect(() => {
        (async () => {
            if (await isConnected()) {
                const res = await freighterApi.getAddress();
                if ('address' in res) setPublicKey(res.address);
            }
        })();
    }, []);

    const connectWallet = async () => {
        try {
            const installed = await isConnected();
            if (!installed.isConnected) {
                alert("Freighter yüklü değil.");
                return;
            }

            const allowedRes = await isAllowed();
            if (!allowedRes.isAllowed) {
                const accessRes = await requestAccess();
                if (!("address" in accessRes)) {
                    alert("Freighter bağlantı izni reddedildi.");
                    console.warn("requestAccess", accessRes);
                    return;
                }
            }

            const addrRes = await getAddress();
            if ("address" in addrRes) {
                setPublicKey(addrRes.address);
            } else {
                console.warn("getAddress başarısız:", addrRes);
            }
        } catch (err) {
            console.error("Freighter hatası:", err);
        }
    };

    return (
        <WalletContext.Provider value={{ publicKey, connectWallet }}>
            {children}
        </WalletContext.Provider>
    );
};

export const useWallet = () => {
    const context = useContext(WalletContext);
    if (!context) throw new Error("useWallet must be used within WalletProvider");
    return context;
};