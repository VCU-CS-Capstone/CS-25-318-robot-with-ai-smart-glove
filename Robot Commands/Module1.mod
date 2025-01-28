MODULE Module1
    !***********************************************************
    !
    ! Module:  Module1
    !
    ! Description:
    !   Program that moves the robot arm based on recieved numbers from a 
    !   TCP connection. 
    ! 
    ! Disclaimer:
    !   This program is meant to be ran in ROBOTSTUDIO only.
    !
    ! Author: Sienna Sterling and Caitlin Ngo
    !
    ! Version: 1.0
    !
    !***********************************************************
    
    
    !***********************************************************
    !
    ! Procedure main
    !
    !   This is the entry point of the program
    !
    !***********************************************************
    
   VAR socketdev server_socket;
   VAR socketdev client_socket;
   VAR string received_str;
   VAR num received_num;
   CONST num MOVE_DISTANCE := 30;
   
   PROC main()
       SocketCreate server_socket;
       SocketBind server_socket, "0.0.0.0", 1025;
       SocketListen server_socket;
       TPWrite "Server listening on port 1025";
       
       SocketAccept server_socket, client_socket;
       TPWrite "Client connected";
       
       WHILE TRUE DO
           SocketReceive client_socket \Str:=received_str;
           TPWrite "Received: " + received_str;
           
           TEST received_str
               CASE "1":
                   received_num := 1;
               CASE "2":
                   received_num := 2;
               CASE "3":
                   received_num := 3;
               CASE "4":
                   received_num := 4;
               CASE "5":
                   received_num := 5;
               CASE "6":
                   received_num := 6;
           ENDTEST
           
           TEST received_num
               CASE 1:
                   MoveL Offs(CRobT(), 0, 0, MOVE_DISTANCE), v100, fine, tool0;
               CASE 2:
                   MoveL Offs(CRobT(), 0, 0, -MOVE_DISTANCE), v100, fine, tool0;
               CASE 3:
                   MoveL Offs(CRobT(), 0, -MOVE_DISTANCE, 0), v100, fine, tool0;
               CASE 4:
                   MoveL Offs(CRobT(), 0, MOVE_DISTANCE, 0), v100, fine, tool0;
               CASE 5:
                   MoveL Offs(CRobT(), -MOVE_DISTANCE, 0, 0), v100, fine, tool0;
               CASE 6:
                   MoveL Offs(CRobT(), MOVE_DISTANCE, 0, 0), v100, fine, tool0;
           ENDTEST
           
       ENDWHILE
       
   ERROR
       TPWrite "Error occurred";
       SocketClose server_socket;
       SocketClose client_socket;
       
   ENDPROC
ENDMODULE
