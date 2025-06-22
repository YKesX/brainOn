import {
    Horizon,          // <-- 'Server' artık Horizon ad-alanında
    Keypair,
    Networks,
    TransactionBuilder,
    Operation,
    Asset,
} from "@stellar/stellar-sdk";

const server = new Horizon.Server("https://horizon.stellar.org");
const NETWORK = Networks.PUBLIC;

export async function sendPayment(
    sourceSecret: string,
    destination: string,
    amount: string
) {
    try {
        const sourceKeypair = Keypair.fromSecret(sourceSecret);
        const account = await server.loadAccount(sourceKeypair.publicKey());

        const tx = new TransactionBuilder(account, {
            fee: (await server.fetchBaseFee()).toString(),
            networkPassphrase: NETWORK,
        })
            .addOperation(
                Operation.payment({
                    destination,
                    asset: Asset.native(),
                    amount,
                })
            )
            .setTimeout(30)
            .build();


        tx.sign(sourceKeypair);
        const result = await server.submitTransaction(tx);
        console.log("✅ Transfer başarılı:", result.hash);
        return result;
    } catch (e: any) {
        console.error("❌ Transfer hatası:", e);
        throw e;
    }
}

/**
 * Cüzdandaki XLM bakiyesini döndürür
 */
export async function fetchBalance(publicKey: string): Promise<string> {
    const account = await server.loadAccount(publicKey);
    const native = account.balances.find(
        (b) => b.asset_type === "native"
    );
    return native ? `${native.balance} XLM` : "0 XLM";
}
