import {
    Horizon,
    Keypair,
    Networks,
    TransactionBuilder,
    Operation,
    Asset,
} from "@stellar/stellar-sdk";


export async function dexBuy(
    sourceSecret: string,
    sellingAsset: Asset,
    buyingAsset: Asset,
    buyAmount: string,
    price: string | { n: number; d: number },
) {
    const server = new Horizon.Server("https://horizon.stellar.org");
    const keypair = Keypair.fromSecret(sourceSecret);
    const account = await server.loadAccount(keypair.publicKey());

    /* ───── 1.  Hesapta buyingAsset için trustline var mı? ───── */
    const needsTrustline =
        !buyingAsset.isNative() &&
        !account.balances.some(
            (b) =>
                "asset_code" in b &&
                "asset_issuer" in b &&
                b.asset_code === buyingAsset.getCode() &&
                b.asset_issuer === buyingAsset.getIssuer(),
        );

    /* ───── 2.  İşlem inşası ───── */
    const builder = new TransactionBuilder(account, {
        fee: (await server.fetchBaseFee()).toString(),
        networkPassphrase: Networks.PUBLIC,
    });

    if (needsTrustline) {
        builder.addOperation(
            Operation.changeTrust({ asset: buyingAsset })   // trustline ekle
        );
    }

    builder.addOperation(
        Operation.manageBuyOffer({
            selling: sellingAsset,
            buying: buyingAsset,
            buyAmount,
            price,
            offerId: "0",          // yeni teklif
        })
    );

    const tx = builder.setTimeout(30).build();

    /* ───── 3.  İmzala & gönder ───── */
    tx.sign(keypair);
    const res = await server.submitTransaction(tx);
    console.log("📈 DEX alım teklifi gönderildi:", res.hash);
    return res;
}
