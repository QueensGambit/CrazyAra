const columnNames = ["A","B","C","D","E","F","G","H"];
const rowNames = [1,2,3,4,5,6,7,8];
const WHITE = true;
const BLACK = false;
const pieces = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"];

const unicode_pieces = {
    "P": "&#9817;", "p": "&#9823;",
    "N": "&#9816;", "n": "&#9822;",
    "B": "&#9815;", "b": "&#9821;",
    "R": "&#9814;", "r": "&#9820;",
    "Q": "&#9813;", "q": "&#9819;",
    "K": "&#9812;", "k": "&#9818;",
};

let board_ui = undefined;
let pocket_ui = undefined;
let promotion_ui = undefined;
let board = undefined;
let pocket = undefined;
let active_square = undefined;
let active_pocket = undefined;
let is_white_to_move = true;

function transpose(a) {
    let result = [];
    for (let i=0; i<a[0].length; i++) {
        let column = Array(a.length);
        for (let j=0; j<a.length; j++) {
            column[j] = a[j][i]
        }
        result.push(column)
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

standard_board = transpose(reverse([
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

function init() {
    board = standard_board;
    pockets = [];
    build_ui();
    present_board(board, pocket);
    showMessage("Updating the board");
    perform_game_update();
}

function build_ui() {
    board_ui = $("#board");
    pocket_ui = $("#pocket");
    promotion_ui = $("#board_promotion");
    let board_labels_x_t = $("#board_labels_x_top");
    let board_labels_x_b = $("#board_labels_x_bottom");
    let board_labels_y_l = $("#board_labels_y_left");
    let board_labels_y_r = $("#board_labels_y_right");

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
                    register_board_click(e, col, row);
                }
            })(col, row));
            board_ui.append(square);
        }
    }

    for (let i = 0; i < 8; i++) {
        let descriptionX = $("<div>").text(columnNames[i]);
        board_labels_x_t.append(descriptionX);
        board_labels_x_b.append(descriptionX.clone());

        let descriptionY = $("<div>").text(rowNames[7-i]);
        board_labels_y_l.append(descriptionY);
        board_labels_y_r.append(descriptionY.clone());
    }


    let promotion_pieces = ["Q", "R", "B", "N"];
    for (let i in promotion_pieces) {
        let piece_str = unicode_pieces[promotion_pieces[i]];
        let square = $("<div>");
        square.addClass("pocket_element").html(piece_str);
        square.click((function(promotion_piece) {
            return function(e) {
                register_board_click(e, -1, -1, undefined, promotion_piece);
            }
        })(promotion_pieces[i]));
        promotion_ui.append(square);
    }

    set_promotion_state(false);
}

let promotion_is_active = false;
let promotion_from = undefined;
let promotion_to = undefined;
function set_promotion_state(active) {
    promotion_is_active = active;
    if (active) {
        promotion_ui.removeClass("greyed_out");
    } else {
        promotion_ui.addClass("greyed_out");
    }
}

function present_board(board, pocket) {
    for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
            let piece = board[x][y];
            let piece_str = "";
            if (piece!=="") {
                piece_str = unicode_pieces[piece];
            }
            $("#"+getUIIdFromColRow(x, y)).html(piece_str);
        }
    }


    for (let i in pocket)  {

        let piece_str = unicode_pieces[pocket[i]];
    }
}

function present_pocket(pockets) {
    let pocketA = $("#pocket_a");
    let pocketB = $("#pocket_b");
    pocketA.empty().html("white<hr>");
    pocketB.empty().html("black<hr>");
    for (let i in pockets[0]) {
        let piece = pockets[0][i].toUpperCase();
        let piece_str = unicode_pieces[piece];

        let uiElement = $("<div>").html(piece_str)
            .addClass("pocket_"+piece)
            .addClass("pocket_element").click((function(piece) {
            return function (e) {
                register_board_click(e, -1, -1, piece);
            }
        })(piece));
        pocketA.append(uiElement);
    }
    for (let i in pockets[1]) {
        let piece = pockets[1][i];
        let piece_str = unicode_pieces[piece];

        let uiElement = $("<div>").html(piece_str)
            .addClass("pocket_"+piece)
            .addClass("pocket_element").click((function(piece) {
            return function (e) {
                register_board_click(e, -1, -1, piece);
            }
        })(piece));
        pocketB.append(uiElement);
    }
}

function showMessage(msg) {
    $("#messages").text(msg);
}

function activate_square(col, row) {
    $(".active_square").removeClass("active_square");

    if (col === -1) {
        active_square = undefined;
        return;
    }

    active_pocket = undefined;
    active_square = [col, row];
    let squareName = getUIIdFromColRow(active_square[0], active_square[1]);
    $("#"+squareName).addClass("active_square");
}

function activate_pocket_piece(piece) {
    $(".active_square").removeClass("active_square");

    if (piece===undefined) {
        active_pocket = undefined;
        return;
    }

    active_square = undefined;
    active_pocket = piece;
    $(".pocket_"+piece).first().addClass("active_square");
}

function getPieceAt(col, row) {
    return board[col][row];
}

function getPieceColor(piece) {
    let pieceChar = piece.charCodeAt(0);
    return ((65<=pieceChar) && (pieceChar<=90));
}


function perform_game_update() {
    $.ajax({
        url: "/api/state",
        success: process_api_response,
        dataType: "json"
    });
}

function perform_new_game() {
    $.ajax({
        url: "/api/new",
        success: process_api_response,
        dataType: "json"
    });
}

let user_action_state = "select_piece";
function register_board_click(e, col, row, drop_piece, promotion_piece) {
    console.log(user_action_state, "->");
    //clicks on the black pocket are ignored
    if ((drop_piece !== undefined) && (!getPieceColor(drop_piece))) {
        return;
    }

    //clicks on the promotion table are ignored, unless we are in select_promotion state
    if ((user_action_state !== "select_promotion") && (promotion_piece !== undefined)) {
        return;
    }

    switch (user_action_state) {
        case "select_piece": {
            if (drop_piece === undefined) {
                let piece = getPieceAt(col, row);
                if ((piece === "") || (!getPieceColor(piece))) {
                    //user clicked on an empty square or on the wrong color
                    return;
                }
                activate_pocket_piece(undefined);
                activate_square(col, row);

                user_action_state = "select_target";
            } else {
                activate_square(-1, -1);
                activate_pocket_piece(drop_piece);
                user_action_state = "select_target";
            }
            break;
        }
        case "select_target": {
            if (drop_piece !== undefined) {
                //user selected another pocket piece.
                user_action_state = "select_piece";
                return register_board_click(e, col, row, drop_piece);
            } else {
                let piece = getPieceAt(col, row);
                if (getPieceColor(piece)) {
                    //user selected another of his pieces.
                    user_action_state = "select_piece";
                    return register_board_click(e, col, row, drop_piece);
                }
            }

            if (active_pocket === undefined) {
                //perform normal move
                //send to server
                let from = getSquareNameFromColRow(active_square[0], active_square[1]);
                let to = getSquareNameFromColRow(col, row);
                let piece = getPieceAt(active_square[0], active_square[1]);
                showMessage("playing " + piece + " " + from + " ->" + to);

                let needs_promotion = ((piece === "P") && (row === 7));
                if (!needs_promotion) {
                    user_action_state = "waiting";

                    $.ajax({
                        url: "/api/move",
                        data: {
                            "from": from,
                            "to": to
                        },
                        success: process_api_response,
                        dataType: "json"
                    });
                } else {
                    //promotion
                    promotion_from = from;
                    promotion_to = to;
                    user_action_state = "select_promotion";

                    set_promotion_state(true);
                }
            } else {
                //perform drop
                let drop = active_pocket.toLowerCase();
                let to = getSquareNameFromColRow(col, row);
                showMessage("dropping " + drop + " @" + to);

                user_action_state = "waiting";

                $.ajax({
                    url: "/api/move",
                    data: {
                        "drop": drop,
                        "to": to
                    },
                    success: process_api_response,
                    dataType: "json"
                });
            }
            break;
        }
        case "select_promotion": {
            set_promotion_state(false);

            user_action_state = "waiting";

            $.ajax({
                    url: "/api/move",
                    data: {
                        "promotion": promotion_piece.toLowerCase(),
                        "from": promotion_from,
                        "to": promotion_to
                    },
                    success: process_api_response,
                    dataType: "json"
                });
            break;
        }
    }
    console.log("->",user_action_state);
}

function api_board_to_matrix(board_str) {
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

function process_api_response(data, status, xhr) {
    activate_square(-1,-1);
    showMessage(data.message);

    board = api_board_to_matrix(data.board);
    pocket = data.pocket.split("|");

    present_board(board);
    present_pocket(pocket);

    if (data.finished===undefined) {
        user_action_state = "select_piece";
    } else {
        user_action_state = "finished";
    }
}
