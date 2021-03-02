const columnNames = ["A","B","C","D","E","F","G","H"];
const rowNames = [1,2,3,4,5,6,7,8];
const WHITE = true;
const BLACK = false;
const pieces = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"];

const unicodePieces = {
    "P": "&#9817;", "p": "&#9823;",
    "N": "&#9816;", "n": "&#9822;",
    "B": "&#9815;", "b": "&#9821;",
    "R": "&#9814;", "r": "&#9820;",
    "Q": "&#9813;", "q": "&#9819;",
    "K": "&#9812;", "k": "&#9818;",
};

let boardUI = undefined;
let pocketUI = undefined;
let promotionUI = undefined;
let board = undefined;
let pocket = undefined;
let activeSquare = undefined;
let activePocket = undefined;

function transpose(a) {
    let result = [];
    for (let i=0; i<a[0].length; i++) {
        let column = Array(a.length);
        for (let j=0; j<a.length; j++) {
            column[j] = a[j][i]
        }
        result.push(column);
    }
    return result;
}

function reverse(a) {
    let result = [];
    for (let i=0; i<a.length; i++) {
        result.push(a[a.length-1-i]);
    }
    return result;
}

standardBoard = transpose(reverse([
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        ["", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ]));

function getSquareNameFromColRow(col, row) {
    return columnNames[col]+rowNames[row];
}

function getUIIdFromColRow(col, row) {
    return "square_"+getSquareNameFromColRow(row, col);
}

function perform_game_update() {
    $.ajax({
        url: "/api/state",
        success: processApiResponse,
        dataType: "json"
    });
}

function init() {
    board = standardBoard;
    buildUI();
    presentBoard(board, pocket);
    showMessage("Updating the board");
    perform_game_update();
}

function buildUI() {
    boardUI = $("#board");
    pocketUI = $("#pocket");
    promotionUI = $("#board_promotion");
    let board_labels_x_t = $("#board_labels_x_top");
    let boardLabelsXB = $("#board_labels_x_bottom");
    let boardLabelsYL = $("#board_labels_y_left");
    let boardLabelsYR = $("#board_labels_y_right");

    for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
            let isWhite = (y + x) % 2 === 0;
            let square = $("<div>");

            let col = x;
            let row = 7-y;

            square.attr("id", getUIIdFromColRow(col, row));
            square.addClass("square");
            square.addClass(isWhite?"white":"black");
            square.click((function(col, row) {
                return function(e) {
                    registerBoardClick(e, col, row);
                }
            })(col, row));
            boardUI.append(square);
        }
    }

    for (let i = 0; i < 8; i++) {
        let descriptionX = $("<div>").text(columnNames[i]);
        board_labels_x_t.append(descriptionX);
        boardLabelsXB.append(descriptionX.clone());

        let descriptionY = $("<div>").text(rowNames[7-i]);
        boardLabelsYL.append(descriptionY);
        boardLabelsYR.append(descriptionY.clone());
    }


    let promotionPieces = ["Q", "R", "B", "N"];
    for (let i in promotionPieces) {
        let piece_str = unicodePieces[promotionPieces[i]];
        let square = $("<div>");
        square.addClass("pocket_element").html(piece_str);
        square.click((function(promotionPiece) {
            return function(e) {
                registerBoardClick(e, -1, -1, undefined, promotionPiece);
            }
        })(promotionPieces[i]));
        promotionUI.append(square);
    }

    setPromotionState(false);
}

let promotionIsActive = false;
let promotionFrom = undefined;
let promotionTo = undefined;
function setPromotionState(active) {
    promotionIsActive = active;
    if (active) {
        promotionUI.removeClass("greyed_out");
    } else {
        promotionUI.addClass("greyed_out");
    }
}

function presentBoard(board, pocket) {
    for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
            let piece = board[x][y];
            let pieceStr = "";
            if (piece!=="") {
                pieceStr = unicodePieces[piece];
            }
            $("#"+getUIIdFromColRow(x, y)).html(pieceStr);
        }
    }


    for (let i in pocket)  {
        let piece_str = unicodePieces[pocket[i]];
    }
}

function getUiElement(piece) {
    let pieceStr = unicodePieces[piece];

    let uiElement = $("<div>").html(pieceStr)
        .addClass("pocket_" + piece)
        .addClass("pocket_element").click((function (piece) {
            return function (e) {
                registerBoardClick(e, -1, -1, piece);
            }
        })(piece));
    return uiElement;
}

function presentPocket(pockets) {
    let pocketA = $("#pocket_a");
    let pocketB = $("#pocket_b");
    pocketA.empty().html("white<hr>");
    pocketB.empty().html("black<hr>");
    for (let i in pockets[0]) {
        let piece = pockets[0][i].toUpperCase();
        let uiElement = getUiElement(piece);
        pocketA.append(uiElement);
    }
    for (let i in pockets[1]) {
        let piece = pockets[1][i];
        let uiElement = getUiElement(piece);
        pocketB.append(uiElement);
    }
}

function showMessage(msg) {
    $("#messages").text(msg);
}

function activateSquare(col, row) {
    $(".active_square").removeClass("active_square");

    if (col === -1) {
        activeSquare = undefined;
        return;
    }

    activePocket = undefined;
    activeSquare = [col, row];
    let squareName = getUIIdFromColRow(activeSquare[0], activeSquare[1]);
    $("#"+squareName).addClass("active_square");
}

function activatePocketPiece(piece) {
    $(".active_square").removeClass("active_square");

    if (piece===undefined) {
        activePocket = undefined;
        return;
    }

    activeSquare = undefined;
    activePocket = piece;
    $(".pocket_"+piece).first().addClass("active_square");
}

function getPieceAt(col, row) {
    return board[col][row];
}

function getPieceColor(piece) {
    let pieceChar = piece.charCodeAt(0);
    return ((65<=pieceChar) && (pieceChar<=90));
}

function perform_new_game() {
    $.ajax({
        url: "/api/new",
        success: processApiResponse,
        dataType: "json"
    });
}

let userActionState = "select_piece";
function registerBoardClick(e, col, row, dropPiece, promotionPiece) {
    console.log(userActionState, "->");
    //clicks on the black pocket are ignored
    if ((dropPiece !== undefined) && (!getPieceColor(dropPiece))) {
        return;
    }

    //clicks on the promotion table are ignored, unless we are in select_promotion state
    if ((userActionState !== "select_promotion") && (promotionPiece !== undefined)) {
        return;
    }

    switch (userActionState) {
        case "select_piece": {
            if (dropPiece === undefined) {
                let piece = getPieceAt(col, row);
                if ((piece === "") || (!getPieceColor(piece))) {
                    //user clicked on an empty square or on the wrong color
                    return;
                }
                activatePocketPiece(undefined);
                activateSquare(col, row);

                userActionState = "select_target";
            } else {
                activateSquare(-1, -1);
                activatePocketPiece(dropPiece);
                userActionState = "select_target";
            }
            break;
        }
        case "select_target": {
            if (dropPiece !== undefined) {
                //user selected another pocket piece.
                userActionState = "select_piece";
                return registerBoardClick(e, col, row, dropPiece);
            } else {
                let piece = getPieceAt(col, row);
                if (getPieceColor(piece)) {
                    //user selected another of his pieces.
                    userActionState = "select_piece";
                    return registerBoardClick(e, col, row, dropPiece);
                }
            }

            if (activePocket === undefined) {
                //perform normal move
                //send to server
                let from = getSquareNameFromColRow(activeSquare[0], activeSquare[1]);
                let to = getSquareNameFromColRow(col, row);
                let piece = getPieceAt(activeSquare[0], activeSquare[1]);
                showMessage("playing " + piece + " " + from + " ->" + to);

                let needs_promotion = ((piece === "P") && (row === 7));
                if (!needs_promotion) {
                    userActionState = "waiting";

                    $.ajax({
                        url: "/api/move",
                        data: {
                            "from": from,
                            "to": to
                        },
                        success: processApiResponse,
                        dataType: "json"
                    });
                } else {
                    //promotion
                    promotionFrom = from;
                    promotionTo = to;
                    userActionState = "select_promotion";

                    setPromotionState(true);
                }
            } else {
                //perform drop
                let drop = activePocket.toLowerCase();
                let to = getSquareNameFromColRow(col, row);
                showMessage("dropping " + drop + " @" + to);

                userActionState = "waiting";

                $.ajax({
                    url: "/api/move",
                    data: {
                        "drop": drop,
                        "to": to
                    },
                    success: processApiResponse,
                    dataType: "json"
                });
            }
            break;
        }
        case "select_promotion": {
            setPromotionState(false);

            userActionState = "waiting";

            $.ajax({
                    url: "/api/move",
                    data: {
                        "promotion": promotionPiece.toLowerCase(),
                        "from": promotionFrom,
                        "to": promotionTo
                    },
                    success: processApiResponse,
                    dataType: "json"
                });
            break;
        }
    }
    console.log("->",userActionState);
}

function apiBoardToMatrix(board_str) {
    let idx = 0;
    let board = [];
    for (let y = 0; y < 8; y++) {
        let row = [];
        for (let x = 0; x < 8; x++) {
            let sq = board_str[idx];
            if (pieces.indexOf(sq)===-1) {
                row.push("");
            } else {
                row.push(sq);
            }
            idx++;
            idx++; //skip '.' or '\n'
        }
        board.push(row);
    }
    return transpose(reverse(board));
}

function processApiResponse(data, status, xhr) {
    activateSquare(-1,-1);
    showMessage(data.message);

    board = apiBoardToMatrix(data.board);
    pocket = data.pocket.split("|");

    presentBoard(board);
    presentPocket(pocket);

    if (data.finished===undefined) {
        userActionState = "select_piece";
    } else {
        userActionState = "finished";
    }
}
