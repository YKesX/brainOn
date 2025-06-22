import { io } from "socket.io-client";

export const socket = io("http://192.168.229.84:5000");


export type EEGAction =
    | "connect_wallet"
    | "8988fb4fc735b6dc5d3b0acad50edf57e5fcf1ff69891940ce2c0ce4490d4ed9"   //   1
    | "a18ac4e6fbd3fc024a07a21dafbac37d828ca8a04a0e34f368f1ec54e0d4fffb"  //  2


    // if it receives class id 0, it will transmit "f7b11509f4d675c3c44f0dd37ca830bb02e8cfa58f04c46283c4bfcbdce1ff45", 
    // if it receives class id 1 it will transmit "8988fb4fc735b6dc5d3b0acad50edf57e5fcf1ff69891940ce2c0ce4490d4ed9" 
    // and if it receives class id 2 it will transmit "a18ac4e6fbd3fc024a07a21dafbac37d828ca8a04a0e34f368f1ec54e0d4fffb".
    
    // 0 hiçbişey
    
    // 1 unlock
    
    // 2 transaction