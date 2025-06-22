import { createContext, useContext, useEffect, useState } from "react";
import { socket, type EEGAction } from "../config/socket";
import toast from "react-hot-toast";
import { fetchBalance, sendPayment } from "../utils/walletUtils";
import { dexBuy } from "../utils/dexPayment"
import { Asset } from "@stellar/stellar-sdk";

type EEGEvent = {
    action: EEGAction;
    signal_raw: string;
    timestamp: string;
    latestTx: string
};

interface EEGContextType {

    isSendLoading: boolean;
    isDexSendLoading: boolean
    balance: string | null;
    latestTx: string
}

const EEGContext = createContext<EEGContextType>({

    isSendLoading: false,
    isDexSendLoading: false,
    balance: null,
    latestTx: ""
});


interface ClassificationEvt {
    classification?: { mode?: string; confidence?: number };
}

export const EEGProvider = ({ children }: { children: React.ReactNode }) => {

    const [latestEvent, setLatestEvent] = useState<EEGEvent>();
    const [log, setLog] = useState<string[]>([]);
    const [isSendLoading, setIsSendLoading] = useState(false);
    const [isDexSendLoading, setIsDexSendLoading] = useState(false);

    const [balance, setBalance] = useState<string | null>(null);

    const [latestTx, setLatestTx] = useState("")

    useEffect(() => {
        interface ClassificationEvt {
            classification?: { mode?: string; confidence?: number };
        }

        const handleClassification = async (data: ClassificationEvt) => {
            const hash = data.classification?.mode;
            if (!hash) return;

            switch (hash) {

                case "8988fb4fc735b6dc5d3b0acad50edf57e5fcf1ff69891940ce2c0ce4490d4ed9":
                    console.log("TX SENT")
                    await handleSend();
                    break;


                case "a18ac4e6fbd3fc024a07a21dafbac37d828ca8a04a0e34f368f1ec54e0d4fffb":
                    await sendDexTx()
                    break;


                default:
                    console.log("Final hash:", hash);
            }
        };

        socket.on("classification", handleClassification);

        return () => {
            socket.off("classification", handleClassification); // cleanup
        };
    }, []);


    const handleSend = async () => {
        try {
            setIsSendLoading(true);

            const tx = await sendPayment(
                "PRIVATE_KEY",
                "GB3JDFW5TXE4TBWWN75J7AJHG5FBP74J35CO33KQ5QTMMX3KIBNO7JRA",
                "2.55"
            );
            setLatestTx(tx.hash)

            toast.success("✅ Payment successfully sent!");

            const newBalance = await fetchBalance(
                "GB3U4B7MURE25RDOODZLSRSMIOIQNMGI2GDHX7HGBSVW4T74XWXTGFOK"
            );
            setBalance(newBalance);
        } catch (e: any) {
            toast.error("❌ Something went wrong!");
            console.error(e);
        } finally {
            setIsSendLoading(false);
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
            const res = await dexBuy(SOURCE_SECRET, sellingAsset, buyingAsset, buyAmount, "4.35");
            toast.success("✅ Tx successfully sent!");
            console.log(res)
            setLatestTx(res.hash)

            setIsDexSendLoading(false)
        } catch (error) {
            toast.error("❌ Something went wrong!");
            console.error(error);
        }
        finally {
            setIsSendLoading(false);
        }

    }

    return (
        <EEGContext.Provider
            value={{ isSendLoading, balance, latestTx, isDexSendLoading }}
        >
            {children}
        </EEGContext.Provider>
    );
};

export const useEEG = () => useContext(EEGContext);

/**socket.on("class_id", (data: any) => {
            console.log("******************************")
            console.log("***********************")

            console.log(data)
            const logEntry = `Sinyal: ${data.signal_raw} → Aksiyon: ${data.action}`;
            console.log(logEntry);
            setLatestEvent(data);
            setLog((prev) => [logEntry, ...prev.slice(0, 9)]);

            switch (data.action) {
                case "a18ac4e6fbd3fc024a07a21dafbac37d828ca8a04a0e34f368f1ec54e0d4fffb":   //dex 
                    break;
                case "8988fb4fc735b6dc5d3b0acad50edf57e5fcf1ff69891940ce2c0ce4490d4ed9":  // transfer
                    handleSend();
                    break;

            }
        });

        socket.on("classification", (data: any) => {
            console.log("EEG action →", data.classification.class, data.classification.confidence);
        }); */