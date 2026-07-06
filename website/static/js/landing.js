document.addEventListener("DOMContentLoaded", function () {


    // =====================================================
    // ELEMENTS
    // =====================================================

    const navbar =
        document.querySelector(".navbar");


    const desktopNavLinks =
        document.querySelectorAll(".nav-link");


    const mobileMenuButton =
        document.getElementById("mobileMenuButton");


    const mobileMenu =
        document.getElementById("mobileMenu");


    const mobileNavLinks =
        document.querySelectorAll(".mobile-nav-link");


    const sectionIds = [

        "home",

        "tentang-stroke",

        "gejala",

        "faktor-risiko",

        "pencegahan",

        "penanganan"

    ];



    // =====================================================
    // MOBILE MENU
    // =====================================================

    mobileMenuButton.addEventListener("click", function () {


        mobileMenu.classList.toggle("open");


        const icon =
            mobileMenuButton.querySelector(
                ".material-symbols-outlined"
            );


        if (mobileMenu.classList.contains("open")) {

            icon.textContent = "close";

        } else {

            icon.textContent = "menu";

        }


    });



    // =====================================================
    // SMOOTH SCROLL
    // =====================================================

    const anchorLinks =
        document.querySelectorAll('a[href^="#"]');


    anchorLinks.forEach(function (anchor) {


        anchor.addEventListener("click", function (event) {


            const targetId =
                this.getAttribute("href");


            if (targetId === "#") {

                return;

            }


            const targetElement =
                document.querySelector(targetId);


            if (!targetElement) {

                return;

            }


            event.preventDefault();


            const navbarHeight =
                navbar.offsetHeight;


            const targetPosition =

                targetElement.offsetTop

                - navbarHeight;


            window.scrollTo({

                top: targetPosition,

                behavior: "smooth"

            });



            // CLOSE MOBILE MENU


            mobileMenu.classList.remove("open");


            const icon =

                mobileMenuButton.querySelector(

                    ".material-symbols-outlined"

                );


            icon.textContent = "menu";


        });


    });



    // =====================================================
    // UPDATE ACTIVE NAVIGATION
    // =====================================================

    function updateActiveNavigation() {


        const scrollPosition =

            window.scrollY

            + navbar.offsetHeight

            + 120;



        let currentSection = "home";



        sectionIds.forEach(function (sectionId) {


            const section =

                document.getElementById(sectionId);



            if (

                section

                &&

                scrollPosition >= section.offsetTop

            ) {


                currentSection = sectionId;


            }


        });



        desktopNavLinks.forEach(function (link) {


            link.classList.remove("active");


            const href =

                link.getAttribute("href");



            if (href === "#" + currentSection) {


                link.classList.add("active");


            }


        });


    }



    // =====================================================
    // NAVBAR SHADOW
    // =====================================================

    function updateNavbarShadow() {


        if (window.scrollY > 20) {


            navbar.classList.add("scrolled");


        } else {


            navbar.classList.remove("scrolled");


        }


    }



    // =====================================================
    // SCROLL EVENT
    // =====================================================

    window.addEventListener("scroll", function () {


        updateNavbarShadow();


        updateActiveNavigation();


    });



    // =====================================================
    // WINDOW RESIZE
    // =====================================================

    window.addEventListener("resize", function () {


        if (window.innerWidth > 1100) {


            mobileMenu.classList.remove("open");


            const icon =

                mobileMenuButton.querySelector(

                    ".material-symbols-outlined"

                );


            icon.textContent = "menu";


        }


    });



    // =====================================================
    // INITIAL STATE
    // =====================================================

    updateNavbarShadow();


    updateActiveNavigation();


});