class TicTacToe{
    constructor(){
        this.board = [];
        this.moves = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]];
    }

    reset(){
        this.board = [];
        for (let i = 0; i < 3; i++){
            let row = [];
            for (let j = 0; j < 3; j++){
                row.push(-1);
            }
            this.board.push(row)
        }
        return this.board;
    }

    step(move, player){
        let d_move = this.moves[move];
        let row = d_move[0];
        let col = d_move[1];
        if (this.board[row][col] == -1){
            this.board[row][col] = player;
            if (this.done(player)) return [this.board, 5, true];
            else if (this.board_is_filled()) return [this.board, 0, true];
            else return [this.board, 1, false];
        }
    }

    done(player){
        let win = NaN;
        let n = this.board.length;

        // check rows
        for (let i = 0; i < n; i++){
            win = true;
            for (let j = 0; j < n; j++){
                if (this.board[i][j] != player){
                    win = false;
                    break;
                }
            }
            if (win) return win;
        }

        // check collumns
        for (let i = 0; i < n; i++){
            win = true;
            for (let j = 0; j < n; j++){
                if (this.board[j][i] != player){
                    win = false;
                    break;
                }
            }
            if (win) return win;
        }
        
        // check diagonals
        win = true;
        for (let i = 0; i < n; i++){
            if (this.board[i][i] != player){
                win = false;
                break;
            }
        }
        if (win) return win;

        win = true;
        for (let i = 0; i < n; i++){
            if (this.board[i][n - 1 - i] != player){
                win = false;
                break;
            }
        }
        if (win) return win;

        for (const row of this.board){
            for (const item of row){
                if (item == -1) return false;
            }
        }
        return true;
    }

    player_lost(player){
        if (player == 1) return this.done(0);
        else return this.done(1);
    }

    board_is_filled(){
        for (const row of this.board){
            for (const item of row){
                if (item == -1) return false;
            }
        }
        return true;
    }

    get_possible_moves(state){
        let flatten_state = state.flat(1);
        let results = [];
        for (let i = 0; i < flatten_state.length; i++){
            if (flatten_state[i] == -1){
                results.push(i);
            }
        }
        return results;
    }

    render(){
        for (const element of this.board){
            console.log(element);
        }
    }
}

class NStepQLearning{
    constructor(env, N, epsilon, play, learning_rate = 0.9, discount_factor = 0.9, initial_q_value = 0.1){
        this.env = env
        this.player = 0
        this.Q = {}  // [string, np.ndarray]
        //this.states = []
        //this.actions = []
        //this.rewards = []
        this.move_history = []
        this.discount_factor = discount_factor
        this.learning_rate = learning_rate
        this.epsilon = epsilon
        this.initial_q_value = initial_q_value
        this.N = N
        this.play = play
    }

    get_q_value(hashed_state){
        if (hashed_state in this.Q) return this.Q[hashed_state];
        else {
            return this.add_initial_q_values(hashed_state);
        }
    }

    get_action(state){
        let q_values = this.get_q_value(state.toString());
        let possible_actions = this.env.get_possible_moves(state);
        let best_possible_action = possible_actions[0];
        for (const element of possible_actions){
            if (q_values[element] > q_values[best_possible_action]) {
                best_possible_action = element;
            }
        }
        return best_possible_action;
    }

    random_epsilon_greedy_policy(state){
        let sample = Math.random();
        if (sample > this.epsilon) return this.get_action(state);
        else {
            let actions = this.env.get_possible_moves(state);
            let hashed_state = state.toString();
            if (hashed_state in this.Q) {
                return actions[Math.floor(Math.random() * actions.length)];
            }
            else {
                this.add_initial_q_values(hashed_state)
                return actions[Math.floor(Math.random() * actions.length)];
            }
        }
    }

    action(state){
        this.player = 0;
        let string_state = state.toString();
        let action = this.random_epsilon_greedy_policy(state);
        let step = this.env.step(action, this.player);
        let next_state = step[0];
        let reward = step[1];
        let done = step[2]
        this.player = 1
        //this.states.push(state.toString());
        //this.actions.push(action);
        //this.rewards.push(reward)
        this.move_history.push([string_state, action])

        if (done)
            this.train(reward);

        state = next_state;
        return [state, action, done];
    }

    train(reward){
        let i = 0;
        let next_max = -1.0;
        this.move_history = this.move_history.reverse();
        for (const h of this.move_history){
            if (next_max === -1.0) {
                this.add_initial_q_values(h[0]);
                this.Q[h[0]][h[1]] = reward;
            }
            else {
                this.add_initial_q_values(h[0]);
                this.Q[h[0]][h[1]] += + this.learning_rate * this.discount_factor * next_max;
            }

            i++;
            next_max = Math.max(...this.Q[h[0]]);
        }
    
        this.reset_agent();
    }

    add_initial_q_values(state){
        if (!(state in this.Q)){
            let q_values = [];
            for (let i = 0; i < 9; i++) q_values.push(this.initial_q_value);
            this.Q[state] = q_values;
            return q_values;
        } 
    }

    reset_agent(){
        this.player = 0;
        this.states = []
        this.actions = []
        this.rewards = []
        this.move_history = []
    }

    expert_policy(state, player){
        let possible_moves = this.env.get_possible_moves(state);

        // First, check if we can win in the next move
        for (const element of possible_moves){
            let d_move = this.env.moves[element];
            let row = d_move[0];
            let col = d_move[1];
            this.env.board[row][col] = 1;
            if (this.env.done(1)){
                this.env.board[row][col] = -1;
                return element;
            }
            else this.env.board[row][col] = -1;
        }

        // Check if the player could win on their next move, and block them
        for (const element of possible_moves){
            let d_move = this.env.moves[element];
            let row = d_move[0];
            let col = d_move[1];
            this.env.board[row][col] = 0;
            if (this.env.done(0)){
                this.env.board[row][col] = -1;
                return element;
            }
            else this.env.board[row][col] = -1;
        }

        if (possible_moves.includes(4)) return 4;

        let list_of_free_corners = [0, 2, 6, 8].filter(value => possible_moves.includes(value));
        if (list_of_free_corners.length != 0) return list_of_free_corners[Math.floor(Math.random()*list_of_free_corners.length)];

        return possible_moves[Math.floor(Math.random()*possible_moves.length)];
    }
}

let env = new TicTacToe();
let agent = new NStepQLearning(env, 3, 0.1, true);
let state = env.reset();

window.addEventListener('DOMContentLoaded', () => {
    const tiles = Array.from(document.querySelectorAll('.tile'));
    const playerDisplay = document.querySelector('.display-player');
    const resetButton = document.querySelector('#reset');
    const announcer = document.querySelector('.announcer');

    const PLAYERX_WON = 'PLAYERX_WON';
    const PLAYERO_WON = 'PLAYERO_WON';
    const TIE = 'TIE';
    let currentPlayer = 'X';
    let isGameActive = true;

    const announce = (type) => {
        switch(type){
            case PLAYERO_WON:
                announcer.innerHTML = 'Player <span class="playerO">O</span> Won';
                break;
            case PLAYERX_WON:
                announcer.innerHTML = 'Player <span class="playerX">X</span> Won';
                break;
            case TIE:
                announcer.innerText = 'Tie';
        }
        announcer.classList.remove('hide');
    };

    const isValidAction = (tile) => {
        return !(tile.innerText === 'X' || tile.innerText === 'O');
    };

    const changePlayer = () => {
        playerDisplay.classList.remove(`player${currentPlayer}`);
        currentPlayer = currentPlayer === 'X' ? 'O' : 'X';
        playerDisplay.innerText = currentPlayer;
        playerDisplay.classList.add(`player${currentPlayer}`);
    }

    const userAction = (tile, index) => {
        if(isValidAction(tile) && isGameActive) {
            tile.innerText = currentPlayer;
            tile.classList.add(`player${currentPlayer}`);

            let d_move = env.moves[index];
            let row = d_move[0];
            let col = d_move[1];
            env.board[row][col] = 1;

            // this player won
            if (env.done(1)){
                agent.train(-5);
                announce(currentPlayer === 'X' ? PLAYERX_WON : PLAYERO_WON);
                isGameActive = false;
            }
            else {
                changePlayer();
                let results = agent.action(state);
                state = results[0];
                let action = results[1];
                let done = results[2];

                tiles[action].innerText = currentPlayer;
                tiles[action].classList.add(`player${currentPlayer}`);

                // the agent won
                if (done){
                    announce(currentPlayer === 'X' ? PLAYERX_WON : PLAYERO_WON);
                    isGameActive = false;
                }
            }

            changePlayer();
        }
    }
    
    const resetBoard = () => {
        env.reset();
        announcer.classList.add('hide');

        if (currentPlayer === 'O') {
            changePlayer();
        }

        tiles.forEach(tile => {
            tile.innerText = '';
            tile.classList.remove('playerX');
            tile.classList.remove('playerO');
        });
    }

    tiles.forEach( (tile, index) => {
        tile.addEventListener('click', () => userAction(tile, index));
    });

    resetButton.addEventListener('click', () => {
        resetBoard();
        state = env.reset();
        isGameActive = true;
    });
});


// https://dev.to/javascriptacademy/create-a-simple-tic-tac-toe-game-using-html-css-javascript-i4k