/*
Plan 2 Automatic Capital Allocation Engine
Single source of truth imported by index.html and aligned with capital_execution_test.py.
*/

function autoPick(row, names) {
    for (const name of names) {
        if (
            row[name] !== undefined &&
            row[name] !== null &&
            String(row[name]).trim() !== ""
        ) {
            return row[name];
        }
    }
    return "";
}

function autoNum(value) {
    const n = Number(
        String(value ?? "")
            .replace(/₹/g, "")
            .replace(/,/g, "")
            .replace(/%/g, "")
            .trim()
    );
    return Number.isFinite(n) ? n : 0;
}

function getAllocationRules(capital) {
    capital = Number(capital);
    if (!Number.isFinite(capital) || capital <= 0) {
        return { targetStocks: 0, maxPerStock: 0, maxPerSector: 3 };
    }

    let targetStocks;
    if (capital < 3000) {
        targetStocks = 2;
    } else if (capital < 5000) {
        targetStocks = 3;
    } else if (capital < 10000) {
        targetStocks = 5;
    } else if (capital < 20000) {
        targetStocks = 8;
    } else if (capital < 30000) {
        targetStocks = 10;
    } else if (capital < 50000) {
        targetStocks = 12;
    } else if (capital < 100000) {
        targetStocks = 15;
    } else {
        targetStocks = 20;
    }

    const maxPerStock = Math.min(
        5000,
        Math.max(1000, Number((capital * 0.075).toFixed(2)))
    );

    return { targetStocks, maxPerStock, maxPerSector: 3 };
}

function calculateAutoAllocation(stocks, capital) {
    capital = Number(capital);

    if (!Number.isFinite(capital) || capital <= 0) {
        return {
            rows: [],
            invested: 0,
            cash: capital || 0,
            stockCount: 0,
            maxPerStock: 0,
            targetStocks: 0,
        };
    }

    const { targetStocks, maxPerStock, maxPerSector } = getAllocationRules(capital);

    const candidates = stocks
        .map((row, index) => {
            const price = autoNum(autoPick(row, ["Live Price", "Price", "Current Price"]));
            const rankRaw = autoPick(row, ["ID Rank", "Rank", "Portfolio Rank", "Momentum Rank"]);
            const rank = rankRaw !== "" ? autoNum(rankRaw) : index + 1;
            const sector = String(autoPick(row, ["Sector"]) || "Other").trim();
            const stock = String(autoPick(row, ["Stock", "Symbol", "Ticker"]) || "").trim();

            return { original: row, stock, sector, price, rank };
        })
        .filter((x) => x.stock && x.price > 0)
        .sort((a, b) => a.rank - b.rank);

    const selected = [];
    const sectorCount = {};
    let invested = 0;

    for (const candidate of candidates) {
        if (selected.length >= targetStocks) {
            break;
        }

        const { stock, sector, price, rank, original } = candidate;

        if (price > maxPerStock) {
            continue;
        }

        if ((sectorCount[sector] || 0) >= maxPerSector) {
            continue;
        }

        const remaining = Number((capital - invested).toFixed(2));
        if (remaining < price) {
            continue;
        }

        const allowed = Math.min(maxPerStock, remaining);
        const shares = Math.floor(allowed / price);
        if (shares < 1) {
            continue;
        }

        const amount = Number((shares * price).toFixed(2));
        if (amount <= 0 || amount > maxPerStock + 1e-9 || amount > remaining + 1e-9) {
            continue;
        }

        selected.push({
            rank,
            stock,
            sector,
            price,
            shares,
            invested: amount,
            weight: Number(((amount / capital) * 100).toFixed(2)),
            momentum: autoPick(original, ["Momentum 12-2", "Momentum %", "Momentum", "Momentum Return"]),
            id: autoPick(original, ["ID", "Id", "ID Score"]),
            signal: autoPick(original, ["Portfolio Signal", "Signal"]),
        });

        invested = Number((invested + amount).toFixed(2));
        sectorCount[sector] = (sectorCount[sector] || 0) + 1;
    }

    selected.sort((a, b) => a.rank - b.rank);

    return {
        rows: selected,
        invested: Number(invested.toFixed(2)),
        cash: Number((capital - invested).toFixed(2)),
        stockCount: selected.length,
        maxPerStock: Number(maxPerStock.toFixed(2)),
        targetStocks,
    };
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { calculateAutoAllocation, getAllocationRules };
}
