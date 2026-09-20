/*
Plan 2 Automatic Capital Allocation Engine

Input:
    Any investment amount

Uses:
    Current Plan 2 Top 50
    Sector-wise classification
    Plan 2 ranking
    Live price
    Whole shares

Rules:
    - Never exceed entered capital
    - Whole shares only
    - Maximum 3 stocks from one sector
    - Higher Plan 2 rank gets priority
    - Prefer sector diversification
    - No stock outside Plan 2 Top 50
    - No broker orders are placed
*/

function calculateAutoAllocation(
    stocks,
    capital
){

    capital = Number(capital);

    if(
        !Number.isFinite(capital) ||
        capital <= 0
    ){
        return {
            rows: [],
            invested: 0,
            cash: capital || 0
        };
    }


    const candidates = stocks
        .map((row,index)=>{

            const price = Number(
                row["Live Price"] ||
                row["Price"] ||
                row["Current Price"] ||
                0
            );

            const rank = Number(
                row["Rank"] ||
                row["ID Rank"] ||
                row["Portfolio Rank"] ||
                index + 1
            );

            const sector =
                String(
                    row["Sector"] ||
                    "Other"
                ).trim();

            const stock =
                String(
                    row["Stock"] ||
                    row["Symbol"] ||
                    row["Ticker"] ||
                    ""
                ).trim();

            return {
                original: row,
                stock,
                sector,
                price,
                rank
            };

        })
        .filter(
            x =>
                x.stock &&
                x.price > 0
        )
        .sort(
            (a,b)=>
                a.rank - b.rank
        );


    /*
    Dynamic target:

    Small capital:
        keep portfolio practical

    Larger capital:
        allow more stocks

    Maximum 3 stocks per sector.
    */

    let targetStocks;

    if(capital < 10000){
        targetStocks = 5;
    }
    else if(capital < 20000){
        targetStocks = 8;
    }
    else if(capital < 30000){
        targetStocks = 10;
    }
    else if(capital < 50000){
        targetStocks = 12;
    }
    else if(capital < 100000){
        targetStocks = 15;
    }
    else{
        targetStocks = 20;
    }


    /*
    Maximum amount per stock.

    For ₹20,000:
        ₹1,500 maximum

    For larger capital:
        increases gradually,
        but never allows one stock
        to dominate the portfolio.
    */

    const maxPerStock =
        Math.min(
            5000,
            Math.max(
                1000,
                capital * 0.075
            )
        );


    const selected = [];
    const sectorCount = {};

    let invested = 0;


    /*
    First pass:
    Plan 2 rank + sector diversification.
    */

    for(
        const candidate of candidates
    ){

        if(
            selected.length >=
            targetStocks
        ){
            break;
        }


        const {
            stock,
            sector,
            price,
            rank,
            original
        } = candidate;


        if(
            sectorCount[sector] >= 3
        ){
            continue;
        }


        const remaining =
            capital - invested;


        if(remaining < price){
            continue;
        }


        const allowed =
            Math.min(
                maxPerStock,
                remaining
            );


        const shares =
            Math.floor(
                allowed / price
            );


        if(shares < 1){
            continue;
        }


        const amount =
            Number(
                (
                    shares * price
                ).toFixed(2)
            );


        if(
            amount <= 0 ||
            amount > remaining
        ){
            continue;
        }


        selected.push({

            rank,

            stock,

            sector,

            price,

            shares,

            invested: amount,

            weight:
                Number(
                    (
                        amount /
                        capital *
                        100
                    ).toFixed(2)
                ),

            momentum:
                original["Momentum %"] ||
                original["Momentum"] ||
                "",

            id:
                original["ID"] ||
                original["Id"] ||
                original["ID Score"] ||
                "",

            signal:
                original["Signal"] ||
                original["Portfolio Signal"] ||
                ""

        });


        invested += amount;


        sectorCount[sector] =
            (
                sectorCount[sector] ||
                0
            ) + 1;
    }


    /*
    Second pass:
    If capital is still available,
    fill unused capacity with existing
    selected sectors only when possible.

    This keeps the allocation practical
    without changing Plan 2 ranking.
    */

    if(
        invested < capital &&
        selected.length < targetStocks
    ){

        for(
            const candidate of candidates
        ){

            if(
                selected.length >=
                targetStocks
            ){
                break;
            }


            const alreadySelected =
                selected.some(
                    x =>
                        x.stock ===
                        candidate.stock
                );


            if(alreadySelected){
                continue;
            }


            const remaining =
                capital - invested;


            if(
                remaining <
                candidate.price
            ){
                continue;
            }


            if(
                (sectorCount[
                    candidate.sector
                ] || 0) >= 3
            ){
                continue;
            }


            const allowed =
                Math.min(
                    maxPerStock,
                    remaining
                );


            const shares =
                Math.floor(
                    allowed /
                    candidate.price
                );


            if(shares < 1){
                continue;
            }


            const amount =
                Number(
                    (
                        shares *
                        candidate.price
                    ).toFixed(2)
                );


            if(
                amount <= 0 ||
                amount > remaining
            ){
                continue;
            }


            selected.push({

                rank:
                    candidate.rank,

                stock:
                    candidate.stock,

                sector:
                    candidate.sector,

                price:
                    candidate.price,

                shares,

                invested:
                    amount,

                weight:
                    Number(
                        (
                            amount /
                            capital *
                            100
                        ).toFixed(2)
                    ),

                momentum:
                    candidate.original[
                        "Momentum %"
                    ] ||
                    candidate.original[
                        "Momentum"
                    ] ||
                    "",

                id:
                    candidate.original[
                        "ID"
                    ] ||
                    candidate.original[
                        "Id"
                    ] ||
                    "",

                signal:
                    candidate.original[
                        "Signal"
                    ] ||
                    candidate.original[
                        "Portfolio Signal"
                    ] ||
                    ""

            });


            invested += amount;


            sectorCount[
                candidate.sector
            ] =
                (
                    sectorCount[
                        candidate.sector
                    ] ||
                    0
                ) + 1;
        }
    }


    /*
    Final ranking remains Plan 2 rank.
    */

    selected.sort(
        (a,b)=>
            a.rank - b.rank
    );


    /*
    Recalculate weights after final selection.
    */

    selected.forEach(row=>{

        row.weight =
            Number(
                (
                    row.invested /
                    capital *
                    100
                ).toFixed(2)
            );

    });


    return {

        rows: selected,

        invested:
            Number(
                invested.toFixed(2)
            ),

        cash:
            Number(
                (
                    capital -
                    invested
                ).toFixed(2)
            ),

        stockCount:
            selected.length,

        maxPerStock:
            Number(
                maxPerStock.toFixed(2)
            ),

        targetStocks

    };
}
