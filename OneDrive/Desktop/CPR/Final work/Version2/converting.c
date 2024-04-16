// Version 1

#define _CRT_SECURE_NO_WARNINGS                                   // Used to suppress any unnecessary warnings
#define BUFFER_SIZE 80                                            // Defining a macro constant with value of 80
#include "converting.h"                                           // Including a header file named "coverting.h"

// Function converts strings to int

void converting(void) {
	printf("*** Start of Converting Strings to int Demo ***\n");
	char intString[BUFFER_SIZE];
	int intNumber;

	// Do while loop for obtaining input from the user

	do {
		printf("Type an int numeric string (q to quit):\n");      
		fgets(intString, BUFFER_SIZE, stdin);                     // Getting input from the user 
        intString[strlen(intString) - 1] = '\0';                  // Removing the newline character from the input
	    if (strcmp(intString, "q") != 0) {                        // Checking whether user wants to quit or not
			intNumber = atoi(intString);                          // String to integer conversion
			printf("Converted number is %d\n", intNumber);
		}
	} while (strcmp(intString, "q") != 0);                        // The loop would iterate until 'q' is obtained as input 
	printf("*** End of Converting Strings to int Demo ***\n\n");
    
    //V2    
    printf("*** Start of Converting Strings to double Demo ***\n");
        char doubleString[BUFFER_SIZE];
        double doubleNumber;
        do{
            printf("Type the double numeric string (q - to quit):\n");
            fgets(doubleString, BUFFER_SIZE, stdin);
            doubleString[strlen(doubleString) - 1] = '\0';
            if((strcmp(doubleString, "q") != 0)){
                doubleNumber = atof(doubleString);
                printf("Converted number is %f\n", doubleNumber);
            }
            
        } while (strcmp(doubleString, "q") != 0);
    printf("*** End of Converting Strings to double Demo ***\n\n");
               

}
